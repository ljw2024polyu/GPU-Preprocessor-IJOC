// Compatibility entry points for recorded --presolver pslp0011 commands.
// Both selector names use the single vendored PSLP 0.0.11 implementation.
#include "presolve/pslp0011_integration.h"
#include "presolve/pslp_integration.h"

bool run_embedded_pslp0011_presolve(const LP_info_cpu* model,
    const HPRLP_parameters* param, LP_info_cpu* reduced_model,
    void** handle, HPRLP_FLOAT* time) {
    return run_embedded_pslp_presolve(model, param, reduced_model, handle, time);
}
bool apply_embedded_pslp0011_postsolve(HPRLP_results* result, void* handle,
    int original_m, int original_n) {
    return apply_embedded_pslp_postsolve(result, handle, original_m, original_n);
}
bool pslp0011_postsolve_and_validate_original_kkt(HPRLP_results* result,
    const LP_info_cpu* original, void* handle, const HPRLP_parameters* param) {
    return postsolve_and_validate_original_kkt(result, original, handle, param);
}
void free_embedded_pslp0011_presolver(void* handle) {
    free_embedded_pslp_presolver(handle);
}
