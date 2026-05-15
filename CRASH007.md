
Symptom: crash in test_sepfir2d_strided_2 on Windows

## Code paths in test_spline_large_2d

`test_spline_large_2d` exercises two independent code paths:

**Path 1 — C fitpack (via `RectBivariateSpline`):**
- Construction: `RectBivariateSpline(x, y, z, s=s)` → `_fitpack.regrid` → C `regrid` → `fpregr` → `fpgrre` → ...
- Evaluation: `spl(x, y)` → `_fitpack.bispev` → C `bispev` → `fpbisp` → `fpbspl`

**Path 2 — pure Python reimplementation (via `_regrid`):**
- Construction: `_regrid(x, y, z, s=s)` → `_regrid_fitpack` in `scipy/interpolate/_regrid.py`
  — this is a pure Python reimplementation of the FITPACK algorithm; it does NOT call C `regrid`
- Evaluation: `_ndbspline_call_like_bivariate(spl_custom, x, y)` → `NdBSpline.__call__` in `scipy/interpolate/_ndbspline.py:136`

`test_spline_large_2d_fuzz` only covers Path 1 (C fitpack). Path 2 is less likely to cause a hard crash
(segfault/OOM) since it's pure Python, but it is not exercised by the fuzzer.

https://as-staging.scicftest.com/actions/scipy/scipy/search/?is_trivial__false=1&status__eq=failure&log_contents__contains=FAILED+scipy%5Cinterpolate%5Ctests%5Ctest_fitpack2.py%3A%3ATestRectBivariateSpline%3A%3Atest_spline_large_2d%5Bs_tols1-shape1%5D+-+worker+%27gw1%27+crashed+while+running+%27build-install%2FLib%2Fsite-packages%2Fscipy%2Finterpolate%2Ftests%2Ftest_fitpack2.py%3A%3ATestRectBivariateSpline%3A%3Atest_spline_large_2d%5Bs_tols1-shape1%5D%27


[gw1] win32 -- Python 3.12.10 C:\hostedtoolcache\windows\Python\3.12.10\arm64\python.exe
worker 'gw1' crashed while running 'build-install/Lib/site-packages/scipy/signal/tests/test_bsplines.py::TestSepfir2d::test_sepfir2d_strided_2[numpy]'
FAILED scipy\interpolate\tests\test_fitpack2.py::TestRectBivariateSpline::test_spline_large_2d[s_tols1-shape1] - worker 'gw1' crashed while running 'build-install/Lib/site-packages/scipy/interpolate/tests/test_fitpack2.py::TestRectBivariateSpline::test_spline_large_2d[s_tols1-shape1]'