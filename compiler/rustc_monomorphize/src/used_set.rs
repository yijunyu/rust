//! Cross-crate used-set: the producer ([`emit_used_sets`]) and consumer ([`UsedSet`]) for
//! `-Zdead-fn-emit-used-set` / `-Zdead-fn-used-set`.
//!
//! In its binary-only form (see [`super::dead_fn_elim`]) the pass is a no-op: a binary's
//! monomorphization collector is already exact, so nothing it emits is unreachable. The
//! *cross-crate* form addresses the real slack: a library is compiled before its callers and
//! must conservatively emit its whole `pub` closure, most of which no final binary calls.
//!
//! **Producer** ([`emit_used_sets`], run on the *binary* at `--emit=metadata`, before any
//! codegen): walk the binary's MIR and record every extern `fn` it references. This is the
//! set of a dependency's functions the binary actually reaches — computed without codegen, so
//! it can drive a *first-build* speedup. The binary's monomorphization collector cannot be
//! used here: it filters to local `DefId`s and never lists non-generic extern fns.
//!
//! **Consumer** ([`UsedSet`], run on each *library*): keep only the used-set functions (plus
//! what soundness requires) and drop the rest before `partition()`.
//!
//! **Identity** is the `local_hash` half of [`DefPathHash`] — the def-path fingerprint within
//! a crate, combined with the crate's `StableCrateId`. The probe reads a dependency's *already
//! built* metadata, so it sees that dependency's real `StableCrateId`; the consumer is that
//! same dependency's compilation. Because Cargo builds each dependency exactly once (one
//! `-Cmetadata`), the two agree and a function hashes identically on both sides. (If the probe
//! read one build of a dependency and the consumer were a *different* build with a different
//! `-Cmetadata`, the low-half hash would differ — so the driver must feed the consumer the same
//! artifact the probe read. Under Cargo this holds by construction.) The mangled symbol name
//! could not be used at all: it embeds the crate disambiguator directly.
//!
//! File format (one 16-hex `local_hash` per line, `#`-comments and blanks ignored):
//! ```text
//! # used-set for `deplib` (MIR probe, DefPathHash local_hash)
//! 98912570aa6ef455
//! ```

use std::collections::BTreeMap;
use std::panic;
use std::path::Path;

use rustc_data_structures::fx::{FxHashSet, FxIndexSet};
use rustc_middle::mir::TerminatorKind;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::def_id::{DefId, LOCAL_CRATE};

/// A parsed used-set: the crate-relative def-paths a downstream binary reaches in this crate.
///
/// Keyed on the **crate-relative def-path string** (`def_path(did).to_string_no_crate_verbose()`,
/// e.g. `::f` or `::{impl#0}::bind`), which names an item by its position in the crate's def
/// tree with **no crate name and no disambiguator**. This is invariant across compilations, so
/// the probe (which sees the dependency as an *extern* crate, possibly built with a different
/// `-Cmetadata` than the pruned codegen pass) and the consumer (which sees the same item as
/// *local*) agree. The `DefPathHash::local_hash` was NOT usable here because it folds in the
/// crate's `StableCrateId` (derived from `-Cmetadata`), which differs between Cargo's rmeta/check
/// build of a dependency and its pruned-codegen build. The mangled symbol name is likewise
/// unusable — it embeds the crate disambiguator.
pub(crate) struct UsedSet {
    paths: FxHashSet<String>,
}

impl UsedSet {
    /// Parse a used-set file (one crate-relative def-path per line). Returns `None` (and warns)
    /// if the file cannot be read, so a missing/corrupt used-set degrades to the sound fallback
    /// of keeping the full `pub` closure rather than miscompiling.
    pub(crate) fn load(tcx: TyCtxt<'_>, path: &Path) -> Option<UsedSet> {
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tcx.sess.dcx().warn(format!(
                    "-Z dead-fn-elimination: cannot read used-set file {}: {e}; \
                     keeping full public closure",
                    path.display()
                ));
                return None;
            }
        };
        let paths = contents
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();
        Some(UsedSet { paths })
    }

    /// Is this crate's function in the binary's used-set? Keyed on the crate-relative def-path,
    /// which the probe and this compile agree on regardless of `-Cmetadata` (see the struct docs).
    pub(crate) fn contains(&self, tcx: TyCtxt<'_>, def_id: DefId) -> bool {
        self.paths.contains(&tcx.def_path(def_id).to_string_no_crate_verbose())
    }

    pub(crate) fn len(&self) -> usize {
        self.paths.len()
    }
}

/// Does this item have generic parameters other than lifetimes (i.e. type or const params, in
/// its own generics or an enclosing impl/fn)? Such items are monomorphized per instantiation and
/// are the collector's job. Lifetime-only generics (`impl<'s> ParsedArg<'s>`) produce a single
/// symbol and are eliminable like any non-generic fn.
fn has_non_lifetime_generics(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    let mut generics = tcx.generics_of(def_id);
    loop {
        for param in &generics.own_params {
            if !matches!(param.kind, ty::GenericParamDefKind::Lifetime) {
                return true;
            }
        }
        match generics.parent {
            Some(parent) => generics = tcx.generics_of(parent),
            None => return false,
        }
    }
}

/// The used-set **probe** (component 1). Walks this crate's MIR — available after analysis,
/// *before* codegen — and records every *extern* (non-local) function it references from a
/// call or as a function value. Each such reference is the exact non-generic dependency
/// function this crate reaches; grouped by crate and written as one `<crate>.usedset` file
/// per dependency into `dir`.
///
/// This is the first-build mechanism: it runs on `--emit=metadata` (no codegen), so a later
/// dependency compile can be pruned against the used-set and codegen'd *once*, rather than
/// fully codegen'd and then discarded. It deliberately does **not** use the monomorphization
/// collector, which filters to local `DefId`s and so never lists non-generic extern fns.
pub(crate) fn emit_used_sets(tcx: TyCtxt<'_>, dir: &Path) {
    if let Err(e) = std::fs::create_dir_all(dir) {
        tcx.sess.dcx().warn(format!("-Z dead-fn-emit-used-set: cannot create {}: {e}", dir.display()));
        return;
    }

    // Collect extern fn DefIds referenced across all local MIR bodies.
    //
    // Walk *pre-inlining* MIR (`mir_drops_elaborated_and_const_checked`), NOT `optimized_mir`.
    // At `-O`, the MIR inliner pulls small cross-crate methods (a broad OS/socket API is full
    // of them) into their callers, deleting the `Call` terminator that named the dependency —
    // so `optimized_mir` would make the probe miss exactly the functions we want to record.
    // The drops-elaborated body still has every call. It is a `Steal`, but the probe runs in
    // the `analysis` pass, before `optimized_mir` steals it, so a `.borrow()` is safe.
    let mut externs: FxIndexSet<DefId> = FxIndexSet::default();
    for &local_def_id in tcx.mir_keys(()) {
        let def_id = local_def_id.to_def_id();
        if !tcx.is_mir_available(def_id) {
            continue;
        }
        // Only walk real function/closure bodies. `mir_keys` also contains synthesized bodies
        // (tuple-struct/enum-variant constructors, `DefKind::Ctor`, and `AnonConst`s) for which
        // forcing `mir_drops_elaborated_and_const_checked` ICEs ("can't type-check body of …
        // {constructor#0}"). Those have no user call terminators worth probing anyway.
        use rustc_hir::def::DefKind;
        if !matches!(tcx.def_kind(def_id), DefKind::Fn | DefKind::AssocFn | DefKind::Closure) {
            continue;
        }
        // Collect the extern fn references from one MIR body's terminators and statements.
        let scan = |body: &rustc_middle::mir::Body<'_>, externs: &mut FxIndexSet<DefId>| {
            for bb in body.basic_blocks.iter() {
                if let TerminatorKind::Call { func, .. } = &bb.terminator().kind
                    && let rustc_middle::mir::Operand::Constant(c) = func
                    && let ty::FnDef(callee, _) = c.const_.ty().kind()
                    && !callee.is_local()
                {
                    externs.insert(*callee);
                }
                for stmt in &bb.statements {
                    use rustc_middle::mir::{Rvalue, StatementKind};
                    if let StatementKind::Assign(box (
                        _,
                        Rvalue::Use(op, _) | Rvalue::Cast(_, op, _),
                    )) = &stmt.kind
                        && let rustc_middle::mir::Operand::Constant(c) = op
                        && let ty::FnDef(callee, _) = c.const_.ty().kind()
                        && !callee.is_local()
                    {
                        externs.insert(*callee);
                    }
                }
            }
        };

        // Prefer the *pre-inlining* drops-elaborated body so cross-crate method calls the MIR
        // inliner would fold away are still visible. But by `analysis` time `optimized_mir` may
        // already have *stolen* it for some items (const-eval, coroutine checks). Borrowing a
        // stolen `Steal` panics (an ICE, not cleanly catchable), so check `is_stolen()` first and
        // fall back to `optimized_mir` for those few items. `catch_unwind` is a last-ditch guard.
        let stolen = tcx.mir_drops_elaborated_and_const_checked(local_def_id);
        // `is_stolen` reads scheduling state not tracked by the query system. That is exactly
        // what we want here (a best-effort probe that skips already-consumed bodies); it does not
        // affect the compiled output, only which extern references we happen to record.
        #[allow(rustc::untracked_query_information)]
        let is_stolen = stolen.is_stolen();
        let scanned = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let mut local: FxIndexSet<DefId> = FxIndexSet::default();
            if is_stolen {
                scan(tcx.optimized_mir(def_id), &mut local);
            } else {
                scan(&stolen.borrow(), &mut local);
            }
            local
        }));
        if let Ok(local) = scanned {
            externs.extend(local);
        }
    }

    // Monomorphization-collector pass (binary only). The MIR walk above sees direct calls and
    // fn-values, but NOT a function reached purely through a *generic instantiation* — e.g. a
    // zero-sized fn item passed as a type parameter to a combinator (`nom`'s `opt(section_spec)`),
    // which is then called inside the combinator's *monomorphized* body. That body is codegen'd in
    // this binary, not in the dependency's own MIR, so only the collector — which walks
    // monomorphized reachability from the entry point — observes `section_spec`. Running it here
    // (on the binary's frontend, no codegen) records exactly those extern fns and closes the
    // indirect-reachability frontier. Guarded to binaries: only they have an entry-point root.
    if tcx.entry_fn(()).is_some() {
        let collected = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let (mono_items, _usage) = crate::collector::collect_crate_mono_items(
                tcx,
                crate::collector::MonoItemCollectionStrategy::Lazy,
            );
            let mut out: FxIndexSet<DefId> = FxIndexSet::default();
            for item in mono_items {
                if let rustc_middle::mono::MonoItem::Fn(instance) = item {
                    let did = instance.def_id();
                    if !did.is_local() {
                        out.insert(did);
                    }
                }
            }
            out
        }));
        if let Ok(collected) = collected {
            externs.extend(collected);
        }
    }

    // Group by defining crate; key each entry by the extern fn's **crate-relative def-path**
    // (`to_string_no_crate_verbose`: no crate name, no disambiguator). This is invariant across
    // compilations, so the consumer — which may be a *different* `-Cmetadata` build of the same
    // dependency (Cargo's rmeta/check unit vs its pruned-codegen unit) — computes the same key.
    let mut per_crate: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    for &did in &externs {
        use rustc_hir::def::DefKind;
        if did.krate == LOCAL_CRATE {
            continue;
        }
        // Record free functions *and inherent methods* — both are eliminable when unused
        // (`Socket::bind` and friends are a broad crate's real surface). Skip trait-impl
        // methods: the consumer's soundness floor never drops them (they are reachable via a
        // trait bound / `dyn` / fn-pointer this crate's MIR can't see), so recording them is
        // pointless. An inherent `AssocFn` has no `trait_item_def_id`.
        let is_eliminable_kind = match tcx.def_kind(did) {
            DefKind::Fn => true,
            DefKind::AssocFn => tcx.associated_item(did).trait_item_def_id().is_none(),
            _ => false,
        };
        if !is_eliminable_kind {
            continue;
        }
        // Skip fns generic over *types or consts* — those are monomorphized per instantiation and
        // handled by the collector, not eliminable as a single symbol. But *lifetime*-only generics
        // (e.g. an `impl<'s> ParsedArg<'s>` method like `to_long`) produce exactly one symbol and
        // MUST be recorded — excluding them silently drops real cross-crate calls.
        if has_non_lifetime_generics(tcx, did) {
            continue;
        }
        let name = tcx.crate_name(did.krate).to_string();
        let def_path = tcx.def_path(did).to_string_no_crate_verbose();
        per_crate.entry(name).or_default().insert(def_path);
    }

    // The used-set of a dependency is the *union* over every crate that calls it: `Main`'s
    // probe names `a::f`, but `b::used` is called from inside `a`'s body, so only `a`'s probe
    // names it. Each crate therefore *appends* its references; duplicate lines are harmless
    // (the consumer dedups into a set). Append is used rather than read-modify-write so the
    // parallel crate compilations writing the same file do not race.
    use std::io::Write;
    let mut wrote = 0;
    for (crate_name, paths) in &per_crate {
        let path = dir.join(format!("{crate_name}.usedset"));
        let mut body = String::new();
        for p in paths {
            body.push_str(p);
            body.push('\n');
        }
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path)
            && f.write_all(body.as_bytes()).is_ok()
        {
            wrote += 1;
        }
    }
    tcx.sess.dcx().note(format!(
        "-Z dead-fn-emit-used-set: probed {} extern fns → {} used-set files in {}",
        externs.len(),
        wrote,
        dir.display()
    ));

    // The binary's probe is the *last* contributor to every dependency's used-set: it runs the
    // monomorphization collector, which is the only pass that sees functions reached purely
    // through generic instantiation (`nom`'s `opt(section_spec)`). Once it has appended those, a
    // dependency's used-set is complete. Signal that by creating a per-crate `<crate>.done` marker
    // (atomically: write a temp then rename). A consumer prunes only if its `.done` exists — else
    // it might read a used-set the collector has not yet finished, and drop a live function. This
    // is a per-crate readiness flag, NOT a global barrier: a consumer that finds no marker keeps
    // its full closure and proceeds, so no job ever blocks (no deadlock).
    if tcx.entry_fn(()).is_some() {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "usedset")
                    && let Some(stem) = p.file_stem().and_then(|s| s.to_str())
                {
                    let done = dir.join(format!("{stem}.done"));
                    let tmp = dir.join(format!("{stem}.done.tmp"));
                    if std::fs::write(&tmp, b"done").is_ok() {
                        let _ = std::fs::rename(&tmp, &done);
                    }
                }
            }
        }
    }
}
