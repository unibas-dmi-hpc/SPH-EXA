#ifndef H5Z_ZFP_PROPS_H
#define H5Z_ZFP_PROPS_H

#ifdef __cplusplus
extern "C" {
#endif

extern herr_t H5Pset_zfp_rate(hid_t plist, double rate); 
extern herr_t H5Pset_zfp_precision(hid_t plist, unsigned int prec); 
extern herr_t H5Pset_zfp_accuracy(hid_t plist, double acc); 
extern herr_t H5Pset_zfp_expert(hid_t plist, unsigned int minbits, unsigned int maxbits,
    unsigned int maxprec, int minexp); 
extern herr_t H5Pset_zfp_reversible(hid_t plist); 

extern void H5Pset_zfp_rate_cdata_f(double rate, size_t *cd_nelmts, unsigned int *cd_values);
extern void H5Pset_zfp_precision_cdata_f(unsigned int prec, size_t *cd_nelmts, unsigned int *cd_values);
extern void H5Pset_zfp_accuracy_cdata_f(double acc, size_t *cd_nelmts, unsigned int *cd_values);
extern void H5Pset_zfp_expert_cdata_f(unsigned int minbits, unsigned int maxbits, unsigned int maxprec,
                                      int minexp, size_t *cd_nelmts, unsigned int *cd_values);
extern void H5Pset_zfp_reversible_cdata_f(size_t *cd_nelmts, unsigned int *cd_values);

#ifdef __cplusplus
}
#endif

#endif
