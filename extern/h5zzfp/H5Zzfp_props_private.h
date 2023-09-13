#ifndef H5Z_ZFP_PROPS_PRIVATE_H
#define H5Z_ZFP_PROPS_PRIVATE_H

typedef struct _h5z_zfp_controls_t {
    unsigned int mode;
    union {
        double rate;
        double acc;
        unsigned int prec;
        struct expert_ {
            unsigned int minbits;
            unsigned int maxbits;
            unsigned int maxprec;
            int minexp;
        } expert;
    } details;
} h5z_zfp_controls_t;

#endif
