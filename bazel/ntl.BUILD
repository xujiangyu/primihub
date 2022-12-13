package(default_visibility = ["//visibility:public",],)

# https://khjtony.github.io/programming/2018/12/31/Add-external-library-to-bazel.html

include_files = [
    "include/NTL/ALL_FEATURES.h",
    "include/NTL/BasicThreadPool.h",
    "include/NTL/FFT.h",
    "include/NTL/FFT_impl.h",
    "include/NTL/FacVec.h",
    "include/NTL/GF2.h",
    "include/NTL/GF2E.h",
    "include/NTL/GF2EX.h",
    "include/NTL/GF2EXFactoring.h",
    "include/NTL/GF2X.h",
    "include/NTL/GF2XFactoring.h",
    "include/NTL/GF2XVec.h",
    "include/NTL/HNF.h",
    "include/NTL/LLL.h",
    "include/NTL/Lazy.h",
    "include/NTL/LazyTable.h",
    "include/NTL/MatPrime.h",
    "include/NTL/PD.h",
    "include/NTL/PackageInfo.h",
    "include/NTL/REPORT_ALL_FEATURES.h",
    "include/NTL/RR.h",
    "include/NTL/SmartPtr.h",
    "include/NTL/WordVector.h",
    "include/NTL/ZZ.h",
    "include/NTL/ZZVec.h",
    "include/NTL/ZZX.h",
    "include/NTL/ZZXFactoring.h",
    "include/NTL/ZZ_limbs.h",
    "include/NTL/ZZ_p.h",
    "include/NTL/ZZ_pE.h",
    "include/NTL/ZZ_pEX.h",
    "include/NTL/ZZ_pEXFactoring.h",
    "include/NTL/ZZ_pX.h",
    "include/NTL/ZZ_pXFactoring.h",
    "include/NTL/ctools.h",
    "include/NTL/fileio.h",
    "include/NTL/linux_s390x.h",
    "include/NTL/lip.h",
    "include/NTL/lzz_p.h",
    "include/NTL/lzz_pE.h",
    "include/NTL/lzz_pEX.h",
    "include/NTL/lzz_pEXFactoring.h",
    "include/NTL/lzz_pX.h",
    "include/NTL/lzz_pXFactoring.h",
    "include/NTL/mat_GF2.h",
    "include/NTL/mat_GF2E.h",
    "include/NTL/mat_RR.h",
    "include/NTL/mat_ZZ.h",
    "include/NTL/mat_ZZ_p.h",
    "include/NTL/mat_ZZ_pE.h",
    "include/NTL/mat_lzz_p.h",
    "include/NTL/mat_lzz_pE.h",
    "include/NTL/mat_poly_ZZ.h",
    "include/NTL/mat_poly_ZZ_p.h",
    "include/NTL/mat_poly_lzz_p.h",
    "include/NTL/matrix.h",
    "include/NTL/new.h",
    "include/NTL/pair.h",
    "include/NTL/pair_GF2EX_long.h",
    "include/NTL/pair_GF2X_long.h",
    "include/NTL/pair_ZZX_long.h",
    "include/NTL/pair_ZZ_pEX_long.h",
    "include/NTL/pair_ZZ_pX_long.h",
    "include/NTL/pair_lzz_pEX_long.h",
    "include/NTL/pair_lzz_pX_long.h",
    "include/NTL/pd_FFT.h",
    "include/NTL/quad_float.h",
    "include/NTL/sp_arith.h",
    "include/NTL/thread.h",
    "include/NTL/tools.h",
    "include/NTL/vec_GF2.h",
    "include/NTL/vec_GF2E.h",
    "include/NTL/vec_GF2XVec.h",
    "include/NTL/vec_RR.h",
    "include/NTL/vec_ZZ.h",
    "include/NTL/vec_ZZVec.h",
    "include/NTL/vec_ZZ_p.h",
    "include/NTL/vec_ZZ_pE.h",
    "include/NTL/vec_double.h",
    "include/NTL/vec_long.h",
    "include/NTL/vec_lzz_p.h",
    "include/NTL/vec_lzz_pE.h",
    "include/NTL/vec_quad_float.h",
    "include/NTL/vec_ulong.h",
    "include/NTL/vec_vec_GF2.h",
    "include/NTL/vec_vec_GF2E.h",
    "include/NTL/vec_vec_RR.h",
    "include/NTL/vec_vec_ZZ.h",
    "include/NTL/vec_vec_ZZ_p.h",
    "include/NTL/vec_vec_ZZ_pE.h",
    "include/NTL/vec_vec_long.h",
    "include/NTL/vec_vec_lzz_p.h",
    "include/NTL/vec_vec_lzz_pE.h",
    "include/NTL/vec_vec_ulong.h",
    "include/NTL/vec_xdouble.h",
    "include/NTL/vector.h",
    "include/NTL/version.h",
    "include/NTL/xdouble.h",
    
    "include/NTL/config.h",
    "include/NTL/mach_desc.h",
    "include/NTL/gmp_aux.h",
]

lib_files = [
    "lib/libntl.a",
]

genrule(
    name = "libntl-build",
    outs = include_files + lib_files,
    cmd = "\n".join([
        'set -x',
        'export INSTALL_DIR=$$(pwd)/$(@D)',
        'export TMP_DIR=$$(mktemp -d -t libntl.XXXXX)',
        'echo $$INSTALL_DIR',
        'echo $$TMP_DIR',
        'mkdir -p $$TMP_DIR',
        'cp -R $$(pwd)/../../../../../external/github_ntl/* $$TMP_DIR',
        'cd $$TMP_DIR',
        'cd src',
        './configure PREFIX=$$INSTALL_DIR NTL_THREADS=on NTL_THREAD_BOOST=on NTL_EXCEPTIONS=on SHARED=on NTL_STD_CXX11=on NTL_SAFE_VECTORS=off TUNE=generic',
        'make -j`nproc` && make install',
	'cp $$TMP_DIR/include/NTL/config.h $$INSTALL_DIR/NTL',
	'cp $$TMP_DIR/include/NTL/mach_desc.h $$INSTALL_DIR/NTL',
	'cp $$TMP_DIR/include/NTL/gmp_aux.h $$INSTALL_DIR/NTL',
    ]),
)

cc_library(
    name = "libntl",
    srcs = lib_files,
    hdrs = include_files,
    # includes=["include"],
    strip_include_prefix = "include",
    # Using an empty include_prefix causes Bazel to emit -I instead of -iquote
    # options for the include directory, so that #include <gmp.h> works.
    include_prefix = "",
    linkstatic = True,
)
