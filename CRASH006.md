# CRASH006: Segfault in Mathieu function ufunc (Python 3.14, Linux)

## Where it crashed

The pytest run segfaulted immediately after `scipy/special/tests/test_trig.py` completed
(output: `..s.` — all 4 tests finished). The next file alphabetically is
`scipy/special/tests/test_ufunc_infra.py`, making it the likely site of the crash —
either during import or the first test.

`test_ufunc_infra.py` contains a single parametrized test (`test_out`) that calls
`scipy.special.mathieu_sem` with `out=` and `where=` keyword arguments.

## C stack trace summary

```
_special_ufuncs.cpython-314-x86_64-linux-gnu.so  +0xa50dc
_special_ufuncs.cpython-314-x86_64-linux-gnu.so  +0xa5fc8
_special_ufuncs.cpython-314-x86_64-linux-gnu.so  +0xa69f4
_special_ufuncs.cpython-314-x86_64-linux-gnu.so  +0x16ced
numpy/_core/_multiarray_umath ...  (ufunc dispatch)
```

Three frames deep inside `_special_ufuncs.so`, called from NumPy's ufunc machinery.

## Conclusion

The segfault is in the **Mathieu ufunc loop** inside `_special_ufuncs.so` —
specifically in `xsf::specfun::mtu0` or a function it calls (`fcoef`, `cem`, `sem`).
This is triggered by `mathieu_sem` in `test_ufunc_infra.py::test_out`.

Note: offset matching is approximate — the reference binary is macOS/CPython 3.13,
the crash binary is Linux/CPython 3.14.

## Files

- `special_ufuncs_symbols.txt` — all symbols from the local `.so`
- `special_ufuncs_text_symbols.txt` — text (code) symbols only, sorted by offset
- `special_ufuncs_near_offsets.txt` — symbols within ±0x2000 of the four crash offsets

# Suspicious runs

https://as-staging.scicftest.com/actions/scipy/scipy/runs/25765742764/job/75677711511/
worker 'gw2' crashed while running 'build-install/usr/lib/python3.12/site-packages/scipy/special/tests/test_ufunc_infra.py::test_out[m_shape5-q_shape5-x_shape5-where_shape5]'
