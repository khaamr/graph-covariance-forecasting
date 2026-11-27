"""Compare our likelihood with the authors' original C calculation.

The user supplies dcc_likestream.c from the university's DCC_NL06.zip. Only
the MATLAB allocation wrapper is replaced; both numerical core functions
remain unchanged. Author source and binaries are not redistributed here.
"""
import argparse
import ctypes
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from dcc_nl import composite_objective, correlation, dcc_backcast, nonlinear_covariance


def verify(source_path):
    source = source_path.read_text()
    source = source[:source.index("/* The gateway routine */")]
    source = source.replace('#include "mex.h"', '').replace('#include "matrix.h"', '')
    shim = '''#include <stdlib.h>
#define mxREAL 0
typedef struct {double *data;} mxArray;
void *mxCalloc(size_t n,size_t s){return calloc(n,s);}
void mxFree(void *p){free(p);}
mxArray *mxCreateDoubleMatrix(size_t a,size_t b,int c){mxArray *x=calloc(1,sizeof(mxArray));x->data=calloc(a*b,sizeof(double));return x;}
double *mxGetPr(mxArray *x){return x->data;}
void mxDestroyArray(mxArray *x){free(x->data);free(x);}
'''
    records = []
    with tempfile.TemporaryDirectory(prefix="dcc-author-check-") as directory:
        folder = Path(directory)
        (folder / "oracle.c").write_text(shim + source)
        flags = ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
        library = folder / "oracle.so"
        subprocess.run(["cc", "-O2", *flags, str(folder / "oracle.c"), "-o", str(library)], check=True)
        oracle = ctypes.CDLL(str(library)).dcc_likestream_core
        pointer = ctypes.POINTER(ctypes.c_double)
        oracle.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_size_t, ctypes.c_size_t,
                           pointer, pointer, pointer, pointer]
        rng = np.random.default_rng(20261127)
        for n_assets, n_dates in [(3, 75), (9, 180), (27, 400)]:
            z = rng.standard_t(8, size=(n_dates, n_assets))
            target, backcast = correlation(nonlinear_covariance(z)), dcc_backcast(z)
            std = np.asfortranarray(np.einsum("ti,tj->ijt", z, z))
            matrices = [std, np.asfortranarray(target), np.asfortranarray(backcast)]
            for alpha, beta in [(.03, .95), (.1, .6), (0., .9), (.2, 0.)]:
                output = ctypes.c_double()
                oracle(alpha, beta, n_assets, n_dates,
                       *(a.ctypes.data_as(pointer) for a in matrices), ctypes.byref(output))
                value = composite_objective(np.array([alpha, beta]), z, target, backcast)[0] * n_dates
                error = abs(value - output.value)
                np.testing.assert_allclose(value, output.value, rtol=1e-12, atol=1e-10)
                records.append({"assets": n_assets, "dates": n_dates, "alpha": alpha,
                                "beta": beta, "absolute_nll_error": error})
    return {"author_c_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "comparisons": records, "matched": True,
            "qualification": "Numerical likelihood-core parity; not a MATLAB end-to-end optimizer comparison."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.author_source)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Matched author C core on", len(result["comparisons"]), "cases")
