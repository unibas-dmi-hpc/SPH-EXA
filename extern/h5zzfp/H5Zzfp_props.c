#include "H5Zzfp_plugin.h"
#include "H5Zzfp_props_private.h"

#include "hdf5.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define H5Z_ZFP_PUSH_AND_GOTO(MAJ, MIN, RET, MSG)     \
do                                                    \
{                                                     \
    H5Epush(H5E_DEFAULT,__FILE__,_funcname_,__LINE__, \
        H5E_ERR_CLS_g,MAJ,MIN,MSG);                   \
    retval = RET;                                     \
    goto done;                                        \
} while(0)

static herr_t H5Pset_zfp(hid_t plist, int mode, ...)
{
    static char const *_funcname_ = "H5Pset_zfp";
    static size_t const ctrls_sz = sizeof(h5z_zfp_controls_t);
    unsigned int flags;
    size_t cd_nelmts = 0;
    unsigned int cd_values[1];
    h5z_zfp_controls_t *ctrls_p = 0;
    int i;
    va_list ap;
    herr_t retval = 0;

    if (0 >= H5Pisa_class(plist, H5P_DATASET_CREATE))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a dataset creation property list class");

    ctrls_p = (h5z_zfp_controls_t *) malloc(ctrls_sz);
    if (0 == ctrls_p)
        H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, -1, "allocation failed for ZFP controls");

    va_start(ap, mode);
    ctrls_p->mode = mode;
    switch (mode)
    {
        case H5Z_ZFP_MODE_RATE:
        {
            ctrls_p->details.rate = va_arg(ap, double);
            if (0 > ctrls_p->details.rate)
                H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADVALUE, -1, "rate out of range.");
            break;
        }
        case H5Z_ZFP_MODE_ACCURACY:
        {
            ctrls_p->details.acc = va_arg(ap, double);
            if (0 > ctrls_p->details.acc)
                H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADVALUE, -1, "accuracy out of range.");
            break;
        }
        case H5Z_ZFP_MODE_PRECISION:
        {
            ctrls_p->details.prec = va_arg(ap, unsigned int);
            break;
        }
        case H5Z_ZFP_MODE_EXPERT:
        {
            ctrls_p->details.expert.minbits = va_arg(ap, unsigned int);
            ctrls_p->details.expert.maxbits = va_arg(ap, unsigned int);
            ctrls_p->details.expert.maxprec = va_arg(ap, unsigned int);
            ctrls_p->details.expert.minexp  = va_arg(ap, int);
            break;
        }
        case H5Z_ZFP_MODE_REVERSIBLE:
        {
            break;
        }
        default:
        {
            H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADVALUE, -1, "bad ZFP mode.");
            break;
        }
    }
    va_end(ap);

    for (i = 0; i < H5Pget_nfilters(plist); i++)
    {
        H5Z_filter_t fid;
        if (0 <= (fid = H5Pget_filter(plist, i, &flags, &cd_nelmts, cd_values, 0, 0, 0))
            && fid == H5Z_FILTER_ZFP)
        {
            if (0 > H5Premove_filter(plist, H5Z_FILTER_ZFP))
                H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, -1, "Unable to remove old ZFP filter from pipeline.");
            break;
        }
    }

    if (0 > H5Pset_filter(plist, H5Z_FILTER_ZFP, H5Z_FLAG_MANDATORY, 0, 0))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, -1, "Unable to put ZFP filter in pipeline.");

    if (0 == H5Pexist(plist, "zfp_controls"))
    {
        retval = H5Pinsert2(plist, "zfp_controls", ctrls_sz, ctrls_p, 0, 0, 0, 0, 0, 0);
    }
    else
    {
        retval = H5Pset(plist, "zfp_controls", ctrls_p);
    }

    /* HDF5 copies the memory we gave it */
    free(ctrls_p);

    return retval;

done:

    if (ctrls_p)
        free(ctrls_p);

    return retval;
}

herr_t H5Pset_zfp_rate(hid_t plist, double rate)
{
    return H5Pset_zfp(plist, H5Z_ZFP_MODE_RATE, rate);
}

herr_t H5Pset_zfp_precision(hid_t plist, unsigned int prec)
{
    return H5Pset_zfp(plist, H5Z_ZFP_MODE_PRECISION, prec);
}

herr_t H5Pset_zfp_accuracy(hid_t plist, double acc)
{
    return H5Pset_zfp(plist, H5Z_ZFP_MODE_ACCURACY, acc);
}

herr_t H5Pset_zfp_expert(hid_t plist, unsigned int minbits, unsigned int maxbits,
    unsigned int maxprec, int minexp)
{
    return H5Pset_zfp(plist, H5Z_ZFP_MODE_EXPERT, minbits, maxbits, maxprec, minexp);
}

herr_t H5Pset_zfp_reversible(hid_t plist)
{
    return H5Pset_zfp(plist, H5Z_ZFP_MODE_REVERSIBLE);
}


/* Used only for Fortran wrappers */

void H5Pset_zfp_rate_cdata_f(double rate, size_t *cd_nelmts_f, unsigned int *cd_values) {
  size_t cd_nelmts = *cd_nelmts_f;
  H5Pset_zfp_rate_cdata(rate, cd_nelmts, cd_values);
  *cd_nelmts_f = cd_nelmts;
}

void H5Pset_zfp_precision_cdata_f(unsigned int prec, size_t *cd_nelmts_f, unsigned int *cd_values) {
  size_t cd_nelmts = *cd_nelmts_f;
  H5Pset_zfp_precision_cdata(prec, cd_nelmts, cd_values);
  *cd_nelmts_f = cd_nelmts;
}

void H5Pset_zfp_accuracy_cdata_f(double acc, size_t *cd_nelmts_f, unsigned int *cd_values) {
  size_t cd_nelmts = *cd_nelmts_f;
  H5Pset_zfp_accuracy_cdata(acc, cd_nelmts, cd_values);
  *cd_nelmts_f = cd_nelmts;
}

void H5Pset_zfp_expert_cdata_f(unsigned int minbits, unsigned int maxbits, unsigned int maxprec, 
                               int minexp, size_t *cd_nelmts_f, unsigned int *cd_values) {
  size_t cd_nelmts = *cd_nelmts_f;
  H5Pset_zfp_expert_cdata(minbits, maxbits, maxprec, minexp, cd_nelmts, cd_values);
  *cd_nelmts_f = cd_nelmts;
}

void H5Pset_zfp_reversible_cdata_f(size_t *cd_nelmts_f, unsigned int *cd_values) {
  size_t cd_nelmts = *cd_nelmts_f;
  H5Pset_zfp_reversible_cdata(cd_nelmts, cd_values);
  *cd_nelmts_f = cd_nelmts;
}

