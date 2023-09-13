#include <stdlib.h>
#include <string.h>

/*
This code was based heavily on one of the HDF5 library's internal
filter, H5Zszip.c. The intention in so doing wasn't so much to 
plagerize HDF5 developers as it was to produce a code that, if
The HDF Group ever decided to in the future, could be easily
integrated with the existing HDF5 library code base.

The logic here for 'Z' and 'B' macros as well as there use within
the code to call ZFP library methods is due to this filter being
part of the Silo library but also supported as a stand-alone
package. In Silo, the ZFP library is embedded inside a C struct
to avoid pollution of the global namespace as well as collision
with any other implementation of ZFP a Silo executable may be
linked with. Calls to ZFP lib methods are preface with 'Z ' 
and calls to bitstream methods with 'B ' as in

    Z zfp_stream_open(...);
    B stream_open(...);
*/

#ifdef Z
#undef Z
#endif

#ifdef B
#undef B
#endif

#ifdef AS_SILO_BUILTIN /* [ */
#include "hdf5.h"
#define USE_C_STRUCTSPACE
#include "zfp.h"
#define Z zfp.
#define B zfpbs.
#else /* ] AS_SILO_BUILTIN [ */
#include "H5PLextern.h"
#include "H5Spublic.h"
#include "zfp.h"
#define Z
#define B 
#endif /* ] AS_SILO_BUILTIN */

#include "H5Zzfp_plugin.h"
#include "H5Zzfp_props_private.h"

/* Convenient CPP logic to capture ZFP lib version numbers as compile time hex number */
#define ZFP_VERSION_NO__(Maj,Min,Pat,Twk)  (0x ## Maj ## Min ## Pat ## Twk)
#define ZFP_VERSION_NO_(Maj,Min,Pat,Twk)   ZFP_VERSION_NO__(Maj,Min,Pat,Twk)
#if defined(ZFP_VERSION_TWEAK)
#define ZFP_VERSION_NO                 ZFP_VERSION_NO_(ZFP_VERSION_MAJOR,ZFP_VERSION_MINOR,ZFP_VERSION_PATCH,ZFP_VERSION_TWEAK)
#elif defined(ZFP_VERSION_RELEASE)
#define ZFP_VERSION_NO                 ZFP_VERSION_NO_(ZFP_VERSION_MAJOR,ZFP_VERSION_MINOR,ZFP_VERSION_RELEASE,0)
#elif defined(ZFP_VERSION_PATCH)
#define ZFP_VERSION_NO                 ZFP_VERSION_NO_(ZFP_VERSION_MAJOR,ZFP_VERSION_MINOR,ZFP_VERSION_PATCH,0)
#else
#error ZFP LIBRARY VERSION NOT DETECTED
#endif

/* Older versions of ZFP don't define this */
#ifndef ZFP_VERSION_STRING
#define ZFP_VERSION_STR__(Maj,Min,Rel) #Maj "." #Min "." #Rel
#define ZFP_VERSION_STR_(Maj,Min,Rel)  ZFP_VERSION_STR__(Maj,Min,Rel)
#define ZFP_VERSION_STRING             ZFP_VERSION_STR_(ZFP_VERSION_MAJOR,ZFP_VERSION_MINOR,ZFP_VERSION_RELEASE)
#endif

/* Older versions of ZFP don't define this publicly */
#ifndef ZFP_CODEC
#define ZFP_CODEC ZFP_VERSION_MINOR
#endif

/* Convenient CPP logic to capture H5Z_ZFP Filter version numbers as string and hex number */
#define H5Z_FILTER_ZFP_VERSION_STR__(Maj,Min,Pat) #Maj "." #Min "." #Pat
#define H5Z_FILTER_ZFP_VERSION_STR_(Maj,Min,Pat)  H5Z_FILTER_ZFP_VERSION_STR__(Maj,Min,Pat)
#define H5Z_FILTER_ZFP_VERSION_STR                H5Z_FILTER_ZFP_VERSION_STR_(H5Z_FILTER_ZFP_VERSION_MAJOR,H5Z_FILTER_ZFP_VERSION_MINOR,H5Z_FILTER_ZFP_VERSION_PATCH)

#define H5Z_FILTER_ZFP_VERSION_NO__(Maj,Min,Pat)  (0x0 ## Maj ## Min ## Pat)
#define H5Z_FILTER_ZFP_VERSION_NO_(Maj,Min,Pat)   H5Z_FILTER_ZFP_VERSION_NO__(Maj,Min,Pat)
#define H5Z_FILTER_ZFP_VERSION_NO                 H5Z_FILTER_ZFP_VERSION_NO_(H5Z_FILTER_ZFP_VERSION_MAJOR,H5Z_FILTER_ZFP_VERSION_MINOR,H5Z_FILTER_ZFP_VERSION_PATCH)

#define H5Z_ZFP_PUSH_AND_GOTO(MAJ, MIN, RET, MSG)     \
do                                                    \
{                                                     \
    H5Epush(H5E_DEFAULT,__FILE__,_funcname_,__LINE__, \
        H5E_ERR_CLS,MAJ,MIN,MSG);                     \
    retval = RET;                                     \
    goto done;                                        \
} while(0)

static int h5z_zfp_was_registered = 0;

static size_t    H5Z_filter_zfp(unsigned int flags, size_t cd_nelmts,
                                const unsigned int cd_values[],
                                size_t nbytes, size_t *buf_size, void **buf);
static htri_t H5Z_zfp_can_apply(hid_t dcpl_id, hid_t type_id, hid_t space_id);
static herr_t H5Z_zfp_set_local(hid_t dcpl_id, hid_t type_id, hid_t space_id);

const H5Z_class2_t H5Z_ZFP[1] = {{

    H5Z_CLASS_T_VERS,       /* H5Z_class_t version          */
    H5Z_FILTER_ZFP,         /* Filter id number             */
    1,                      /* encoder_present flag         */
    1,                      /* decoder_present flag         */
    "H5Z-ZFP"               /* Filter name for debugging    */
    "-" H5Z_FILTER_ZFP_VERSION_STR
    " (ZFP-" ZFP_VERSION_STRING ")",
    H5Z_zfp_can_apply,      /* The "can apply" callback     */
    H5Z_zfp_set_local,      /* The "set local" callback     */
    H5Z_filter_zfp,         /* The actual filter function   */

}};

#ifdef H5Z_ZFP_AS_LIB
int H5Z_zfp_initialize(void)
{
    if (H5Zfilter_avail(H5Z_FILTER_ZFP))
        return 1;
    if (H5Zregister(H5Z_ZFP)<0)
        return -1;
    h5z_zfp_was_registered = 1;
    return 1;
}
#else
H5PL_type_t H5PLget_plugin_type(void) {return H5PL_TYPE_FILTER;}
const void *H5PLget_plugin_info(void) {return H5Z_ZFP;}
#endif

#ifndef H5Z_ZFP_AS_LIB
static
#endif
int H5Z_zfp_finalize(void)
{
    herr_t ret2 = 0;
    if (h5z_zfp_was_registered)
        ret2 = H5Zunregister(H5Z_FILTER_ZFP);
    h5z_zfp_was_registered = 0;
    if (ret2 < 0) return -1;
    return 1;
}

static htri_t
H5Z_zfp_can_apply(hid_t dcpl_id, hid_t type_id, hid_t chunk_space_id)
{   
    static char const *_funcname_ = "H5Z_zfp_can_apply";
    int const max_ndims = (ZFP_VERSION_NO <= 0x0053) ? 3 : 4;
    int ndims, ndims_used = 0;
    size_t i, dsize;
    htri_t retval = 0;
    hsize_t dims[H5S_MAX_RANK];
    H5T_class_t dclass;
    hid_t native_type_id;

    /* Disable the ZFP filter entirely if it looks like the ZFP library
       hasn't been compiled for 8-bit stream word size */
    if ((int) B stream_word_bits != 8)
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTINIT, -1,
            "ZFP lib not compiled with -DBIT_STREAM_WORD_TYPE=uint8");

    /* get datatype class, size and space dimensions */
    if (H5T_NO_CLASS == (dclass = H5Tget_class(type_id)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, -1, "bad datatype class");

    if (0 == (dsize = H5Tget_size(type_id)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, -1, "bad datatype size");

    if (0 > (ndims = H5Sget_simple_extent_dims(chunk_space_id, dims, 0)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, -1, "bad chunk data space");

    /* confirm ZFP library can handle this data */
#if ZFP_VERSION_NO < 0x0510
    if (!(dclass == H5T_FLOAT))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0,
            "requires datatype class of H5T_FLOAT");
#else
    if (!(dclass == H5T_FLOAT || dclass == H5T_INTEGER))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0,
            "requires datatype class of H5T_FLOAT or H5T_INTEGER");
#endif

    if (!(dsize == 4 || dsize == 8))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0,
            "requires datatype size of 4 or 8");

    /* check for *USED* dimensions of the chunk */
    for (i = 0; i < ndims; i++)
    {
        if (dims[i] <= 1) continue;
        ndims_used++;
    }

    if (ndims_used == 0 || ndims_used > max_ndims)
#if ZFP_VERSION_NO < 0x0530
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0,
            "chunk must have only 1...3 non-unity dimensions");
#else
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0,
            "chunk must have only 1...4 non-unity dimensions");
#endif

    /* if caller is doing "endian targetting", disallow that */
    native_type_id = H5Tget_native_type(type_id, H5T_DIR_ASCEND);
    if (H5Tget_order(type_id) != H5Tget_order(native_type_id))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0,
            "endian targetting non-sensical in conjunction with ZFP filter");

    retval = 1;

done:

    return retval;
}

static herr_t
H5Z_zfp_set_local(hid_t dcpl_id, hid_t type_id, hid_t chunk_space_id)
{   
    static char const *_funcname_ = "H5Z_zfp_set_local";
    int i, ndims, ndims_used = 0;
    size_t dsize, hdr_bits, hdr_bytes;
    size_t mem_cd_nelmts = H5Z_ZFP_CD_NELMTS_MEM;
    unsigned int mem_cd_values[H5Z_ZFP_CD_NELMTS_MEM];
    size_t hdr_cd_nelmts = H5Z_ZFP_CD_NELMTS_MAX;
    unsigned int hdr_cd_values[H5Z_ZFP_CD_NELMTS_MAX];
    unsigned int flags = 0;
    herr_t retval = 0;
    hsize_t dims[H5S_MAX_RANK], dims_used[H5S_MAX_RANK];
    H5T_class_t dclass;
    zfp_type zt;
    zfp_field *dummy_field = 0;
    bitstream *dummy_bstr = 0;
    zfp_stream *dummy_zstr = 0;
    int have_zfp_controls = 0;
    h5z_zfp_controls_t ctrls;

    if (0 > (dclass = H5Tget_class(type_id)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a datatype");

    if (0 == (dsize = H5Tget_size(type_id)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a datatype");

    if (0 > (ndims = H5Sget_simple_extent_dims(chunk_space_id, dims, 0)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a data space");

    /* setup zfp data type for meta header */
    if (dclass == H5T_FLOAT)
    {
        if (dsize == sizeof(float))
            zt = zfp_type_float;
        else if (dsize == sizeof(double))
            zt = zfp_type_double;
        else
            H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "invalid datatype size");
    }
    else if (dclass == H5T_INTEGER)
    {
        if (dsize == sizeof(int32))
            zt = zfp_type_int32;
        else if (dsize == sizeof(int64))
            zt = zfp_type_int64;
        else
            H5Z_ZFP_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "invalid datatype size");
    }
    else
    {
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0,
            "datatype class must be H5T_FLOAT or H5T_INTEGER");
    }

    /* computed used (e.g. non-unity) dimensions in chunk */
    for (i = 0; i < ndims; i++)
    {
        if (dims[i] <= 1) continue;
        dims_used[ndims_used] = dims[i];
        ndims_used++;
    }

    /* set up dummy zfp field to compute meta header */
    switch (ndims_used)
    {
        case 1: dummy_field = Z zfp_field_1d(0, zt, dims_used[0]); break;
        case 2: dummy_field = Z zfp_field_2d(0, zt, dims_used[1], dims_used[0]); break;
        case 3: dummy_field = Z zfp_field_3d(0, zt, dims_used[2], dims_used[1], dims_used[0]); break;
#if ZFP_VERSION_NO >= 0x0540
        case 4: dummy_field = Z zfp_field_4d(0, zt, dims_used[3], dims_used[2], dims_used[1], dims_used[0]); break;
#endif
#if ZFP_VERSION_NO < 0x0530
        default: H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0,
                     "chunks may have only 1...3 non-unity dims");
#else
        default: H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0,
                     "chunks may have only 1...4 non-unity dims");
#endif
    }
    if (!dummy_field)
        H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "zfp_field_Xd() failed");

    /* get current cd_values and re-map to new cd_value set */
    if (0 > H5Pget_filter_by_id(dcpl_id, H5Z_FILTER_ZFP, &flags, &mem_cd_nelmts, mem_cd_values, 0, NULL, NULL))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "unable to get current ZFP cd_values");

    /* Handle default case when no cd_values are passed by using ZFP library defaults. */
    if (mem_cd_nelmts == 0)
    {
        /* check for filter controls in the properites */
        if (0 < H5Pexist(dcpl_id, "zfp_controls"))
        {
            if (0 > H5Pget(dcpl_id, "zfp_controls", &ctrls))
                H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "unable to get ZFP controls");
            have_zfp_controls = 1;
        }
        else /* just use ZFP library defaults */
        {
            mem_cd_nelmts = H5Z_ZFP_CD_NELMTS_MEM;
            H5Pset_zfp_expert_cdata(ZFP_MIN_BITS, ZFP_MAX_BITS, ZFP_MAX_PREC, ZFP_MIN_EXP, mem_cd_nelmts, mem_cd_values);
        }
    }
        
    /* Into hdr_cd_values, we encode ZFP library and H5Z-ZFP plugin version info at
       entry 0 and use remaining entries as a tiny buffer to write ZFP native header. */
    hdr_cd_values[0] = (unsigned int) ((ZFP_VERSION_NO<<16) | (ZFP_CODEC<<12) | H5Z_FILTER_ZFP_VERSION_NO);
    if (0 == (dummy_bstr = B stream_open(&hdr_cd_values[1], sizeof(hdr_cd_values))))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "stream_open() failed");

    if (0 == (dummy_zstr = Z zfp_stream_open(dummy_bstr)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "zfp_stream_open() failed");

    /* Set the ZFP stream mode from zfp_control properties or mem_cd_values[0] */
    if (have_zfp_controls)
    {
        switch (ctrls.mode)
        {
            case H5Z_ZFP_MODE_RATE:
                Z zfp_stream_set_rate(dummy_zstr, ctrls.details.rate, zt, ndims_used, 0);
                break;
            case H5Z_ZFP_MODE_PRECISION:
#if ZFP_VERSION_NO < 0x0510
                Z zfp_stream_set_precision(dummy_zstr, ctrls.details.prec, zt);
#else
                Z zfp_stream_set_precision(dummy_zstr, ctrls.details.prec);
#endif
                break;
            case H5Z_ZFP_MODE_ACCURACY:
#if ZFP_VERSION_NO < 0x0510
                Z zfp_stream_set_accuracy(dummy_zstr, ctrls.details.acc, zt);
#else
                Z zfp_stream_set_accuracy(dummy_zstr, ctrls.details.acc);
#endif
                break;
            case H5Z_ZFP_MODE_EXPERT:
                Z zfp_stream_set_params(dummy_zstr, ctrls.details.expert.minbits,
                    ctrls.details.expert.maxbits, ctrls.details.expert.maxprec,
                    ctrls.details.expert.minexp);
                break;
#if ZFP_VERSION_NO >= 0x0550
            case H5Z_ZFP_MODE_REVERSIBLE:
                Z zfp_stream_set_reversible(dummy_zstr);
                break;
#endif
            default:
                H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0, "invalid ZFP mode");
        }
    }
    else
    {
        switch (mem_cd_values[0])
        {
            case H5Z_ZFP_MODE_RATE:
                Z zfp_stream_set_rate(dummy_zstr, *((double*) &mem_cd_values[2]), zt, ndims_used, 0);
                break;
            case H5Z_ZFP_MODE_PRECISION:
#if ZFP_VERSION_NO < 0x0510
                Z zfp_stream_set_precision(dummy_zstr, mem_cd_values[2], zt);
#else
                Z zfp_stream_set_precision(dummy_zstr, mem_cd_values[2]);
#endif
                break;
            case H5Z_ZFP_MODE_ACCURACY:
#if ZFP_VERSION_NO < 0x0510
                Z zfp_stream_set_accuracy(dummy_zstr, *((double*) &mem_cd_values[2]), zt);
#else
                Z zfp_stream_set_accuracy(dummy_zstr, *((double*) &mem_cd_values[2]));
#endif
                break;
            case H5Z_ZFP_MODE_EXPERT:
                Z zfp_stream_set_params(dummy_zstr, mem_cd_values[2], mem_cd_values[3],
                    mem_cd_values[4], (int) mem_cd_values[5]);
                break;
#if ZFP_VERSION_NO >= 0x0550
            case H5Z_ZFP_MODE_REVERSIBLE:
                Z zfp_stream_set_reversible(dummy_zstr);
                break;
#endif
            default:
                H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0, "invalid ZFP mode");
        }
    }

    /* Use ZFP's write_header method to write the ZFP header into hdr_cd_values array */
    if (0 == (hdr_bits = Z zfp_write_header(dummy_zstr, dummy_field, ZFP_HEADER_FULL)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTINIT, 0, "unable to write header");

    /* Flush the ZFP stream */
    Z zfp_stream_flush(dummy_zstr);

    /* compute necessary hdr_cd_values size */
    hdr_bytes     = 1 + ((hdr_bits  - 1) / 8);
    hdr_cd_nelmts = 1 + ((hdr_bytes - 1) / sizeof(hdr_cd_values[0]));
    hdr_cd_nelmts++; /* for slot 0 holding version info */

    if (hdr_cd_nelmts > H5Z_ZFP_CD_NELMTS_MAX)
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, -1, "buffer overrun in hdr_cd_values");

    /* Now, update cd_values for the filter */
    if (0 > H5Pmodify_filter(dcpl_id, H5Z_FILTER_ZFP, flags, hdr_cd_nelmts, hdr_cd_values))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0,
            "failed to modify cd_values");

    /* cleanup the dummy ZFP stuff we used to generate the header */
    Z zfp_field_free(dummy_field); dummy_field = 0;
    Z zfp_stream_close(dummy_zstr); dummy_zstr = 0;
    B stream_close(dummy_bstr); dummy_bstr = 0;

    retval = 1;

done:

    if (dummy_field) Z zfp_field_free(dummy_field);
    if (dummy_zstr) Z zfp_stream_close(dummy_zstr);
    if (dummy_bstr) B stream_close(dummy_bstr);
    return retval;
}

static int
get_zfp_info_from_cd_values(size_t cd_nelmts, unsigned int const *cd_values,
    uint64 *zfp_mode, uint64 *zfp_meta, H5T_order_t *swap)
{
    static char const *_funcname_ = "get_zfp_info_from_cd_values";
    unsigned int cd_values_copy[H5Z_ZFP_CD_NELMTS_MAX];
    int retval = 0;
    bitstream *bstr = 0;
    zfp_stream *zstr = 0;
    zfp_field *zfld = 0;

    if (cd_nelmts > H5Z_ZFP_CD_NELMTS_MAX)
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_OVERFLOW, 0, "cd_nelmts exceeds max");

    /* make a copy of cd_values in case we need to byte-swap it */
    memcpy(cd_values_copy, cd_values, cd_nelmts * sizeof(cd_values[0]));

    /* treat the cd_values as a zfp bitstream buffer */
    if (0 == (bstr = B stream_open(&cd_values_copy[0], sizeof(cd_values_copy[0]) * cd_nelmts)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "opening header bitstream failed");

    if (0 == (zstr = Z zfp_stream_open(bstr)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "opening header zfp stream failed");

    /* Allocate the field object */
    if (0 == (zfld = Z zfp_field_alloc()))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "allocating field failed");

    /* Do a read of *just* magic to detect possible codec version mismatch */
    if (0 == (Z zfp_read_header(zstr, zfld, ZFP_HEADER_MAGIC)))
    {
        herr_t conv;

        /* The read may have failed due to difference in endian-ness between
           writer and reader. So, byte-swap cd_values array, rewind the stream and re-try. */
        if (H5T_ORDER_LE == (*swap = (H5Tget_order(H5T_NATIVE_UINT))))
            conv = H5Tconvert(H5T_STD_U32BE, H5T_NATIVE_UINT, cd_nelmts, cd_values_copy, 0, H5P_DEFAULT);
        else
            conv = H5Tconvert(H5T_STD_U32LE, H5T_NATIVE_UINT, cd_nelmts, cd_values_copy, 0, H5P_DEFAULT);
        if (conv < 0)
            H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0, "header endian-swap failed");

        Z zfp_stream_rewind(zstr);
        if (0 == (Z zfp_read_header(zstr, zfld, ZFP_HEADER_MAGIC)))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "ZFP codec version mismatch");
    }
    Z zfp_stream_rewind(zstr);

    /* Now, read ZFP *full* header */
    if (0 == (Z zfp_read_header(zstr, zfld, ZFP_HEADER_FULL)))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "reading header failed");

    /* Get ZFP stream mode and field meta */
    *zfp_mode = Z zfp_stream_mode(zstr);
    *zfp_meta = Z zfp_field_metadata(zfld);

    /* cleanup */
    Z zfp_field_free(zfld); zfld = 0;
    Z zfp_stream_close(zstr); zstr = 0;
    B stream_close(bstr); bstr = 0;
    retval = 1;

done:
    if (zfld) Z zfp_field_free(zfld);
    if (zstr) Z zfp_stream_close(zstr);
    if (bstr) B stream_close(bstr);

    return retval;
}

/*
Compare ZFP codec version used when data was written to what is
currently being used to read the data. There is a challenge here
in that earlier versions of this filter recorded only the ZFP
library version, not the codec version. Although ZFP codec version
was encoded as minor digit of ZFP library version, that convention
ended with ZFP version 1.0.0. So, if an old version of this filter
is used with newer ZFP libraries, we won't know the codec version
used to write the data with certainty. The best we can do is guess
it. If there becomes a version of the ZFP library for which that guess
(currently 5) is wrong, the logic here will need to be updated to
capture knowledge of the ZFP library version for which the codec
version was incrimented.
*/

static int
zfp_codec_version_mismatch(
    unsigned int h5zfpver_from_cd_val_data_in_file,
    unsigned int zfpver_from_cd_val_data_in_file,
    unsigned int zfpcodec_from_cd_val_data_in_file)
{
    int writer_codec;
    int reader_codec;

    if (h5zfpver_from_cd_val_data_in_file < 0x0110)
    {
        /* for data written with older versions of the filter,
           we infer codec from ZFP library version stored in the file. */
        zfpver_from_cd_val_data_in_file <<= 4;
        if (zfpver_from_cd_val_data_in_file < 0x0500)
            writer_codec = 4;
        else if (zfpver_from_cd_val_data_in_file < 0x1000)
            writer_codec = (zfpver_from_cd_val_data_in_file & 0x0F00)>>8;
        else if (zfpver_from_cd_val_data_in_file == 0x1000)
            writer_codec = 5;
        else
            writer_codec = 5; /* can only guess */
    }
    else
        writer_codec = zfpcodec_from_cd_val_data_in_file;

#if ZFP_VERSION_NO < 0x0500
    reader_codec = 4;
#elif ZFP_VERSION_NO < 0x1000
    reader_codec = 5;
#else
    reader_codec = ZFP_CODEC;
#endif

    return writer_codec > reader_codec;
}

static size_t
H5Z_filter_zfp(unsigned int flags, size_t cd_nelmts,
    const unsigned int cd_values[], size_t nbytes,
    size_t *buf_size, void **buf)
{
    static char const *_funcname_ = "H5Z_filter_zfp";
    void *newbuf = 0;
    size_t retval = 0;
    unsigned int cd_vals_h5zzfpver = cd_values[0]&0x00000FFF;
    unsigned int cd_vals_zfpcodec = (cd_values[0]>>12)&0x0000000F;
    unsigned int cd_vals_zfpver = (cd_values[0]>>16)&0x0000FFFF;
    H5T_order_t swap = H5T_ORDER_NONE;
    uint64 zfp_mode, zfp_meta;
    bitstream *bstr = 0;
    zfp_stream *zstr = 0;
    zfp_field *zfld = 0;

    /* Pass &cd_values[1] here to strip off first entry holding version info */
    if (0 == get_zfp_info_from_cd_values(cd_nelmts-1, &cd_values[1], &zfp_mode, &zfp_meta, &swap))
        H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "can't get ZFP mode/meta");

    if (flags & H5Z_FLAG_REVERSE) /* decompression */
    {
        int status;
        size_t bsize, dsize;

        /* Worry about zfp version mismatch only for decompression */
        if (zfp_codec_version_mismatch(cd_vals_h5zzfpver, cd_vals_zfpver, cd_vals_zfpcodec))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_READERROR, 0, "ZFP codec version mismatch");

        /* Set up the ZFP field object */
        if (0 == (zfld = Z zfp_field_alloc()))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "field alloc failed");

        Z zfp_field_set_metadata(zfld, zfp_meta);

        bsize = Z zfp_field_size(zfld, 0);
        switch (Z zfp_field_type(zfld))
        {
            case zfp_type_int32:  dsize = sizeof(int32);  break;
            case zfp_type_int64:  dsize = sizeof(int64);  break;
            case zfp_type_float:  dsize = sizeof(float);  break;
            case zfp_type_double: dsize = sizeof(double); break;
            default: H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0, "invalid datatype");
        }
        bsize *= dsize;

        if (NULL == (newbuf = malloc(bsize)))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0,
                "memory allocation failed for ZFP decompression");

        Z zfp_field_set_pointer(zfld, newbuf);

        /* Setup the ZFP stream object */
        if (0 == (bstr = B stream_open(*buf, *buf_size)))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "bitstream open failed");

        if (0 == (zstr = Z zfp_stream_open(bstr)))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "zfp stream open failed");

        Z zfp_stream_set_mode(zstr, zfp_mode);

        /* Do the ZFP decompression operation */
        status = Z zfp_decompress(zstr, zfld);

        /* clean up */
        Z zfp_field_free(zfld); zfld = 0;
        Z zfp_stream_close(zstr); zstr = 0;
        B stream_close(bstr); bstr = 0;

        if (!status)
            H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTFILTER, 0, "decompression failed");

	/* ZFP is an endian-independent format. It will produce correct endian-ness
           during decompress regardless of endian-ness differences between reader 
           and writer. However, the HDF5 library will not be expecting that. So,
           we need to undue the correct endian-ness here. We use HDF5's built-in
           byte-swapping here. Because we know we need only to endian-swap,
           we treat the data as unsigned. */
        if (swap != H5T_ORDER_NONE)
        {
            hid_t src = dsize == 4 ? H5T_STD_U32BE : H5T_STD_U64BE; 
            hid_t dst = dsize == 4 ? H5T_NATIVE_UINT32 : H5T_NATIVE_UINT64;
            if (swap == H5T_ORDER_BE)
                src = dsize == 4 ? H5T_STD_U32LE : H5T_STD_U64LE; 
            if (H5Tconvert(src, dst, bsize/dsize, newbuf, 0, H5P_DEFAULT) < 0)
                H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0, "endian-UN-swap failed");
        }

        free(*buf);
        *buf = newbuf;
        newbuf = 0;
        *buf_size = bsize; 
        retval = bsize;
    }
    else /* compression */
    {
        size_t msize, zsize;

        /* Set up the ZFP field object */
        if (0 == (zfld = Z zfp_field_alloc()))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "field alloc failed");

        Z zfp_field_set_pointer(zfld, *buf);
        Z zfp_field_set_metadata(zfld, zfp_meta);

        /* Set up the ZFP stream object for real compression now */
        if (0 == (zstr = Z zfp_stream_open(0)))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "zfp stream open failed");

        Z zfp_stream_set_mode(zstr, zfp_mode);
        msize = Z zfp_stream_maximum_size(zstr, zfld);

        /* Set up the bitstream object */
        if (NULL == (newbuf = malloc(msize)))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0,
                "memory allocation failed for ZFP compression");

        if (0 == (bstr = B stream_open(newbuf, msize)))
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_NOSPACE, 0, "bitstream open failed");

        Z zfp_stream_set_bit_stream(zstr, bstr);

        /* Do the compression */
        zsize = Z zfp_compress(zstr, zfld);

        /* clean up */
        Z zfp_field_free(zfld); zfld = 0;
        Z zfp_stream_close(zstr); zstr = 0;
        B stream_close(bstr); bstr = 0;

        if (zsize == 0)
            H5Z_ZFP_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTFILTER, 0, "compression failed");

        if (zsize > msize)
            H5Z_ZFP_PUSH_AND_GOTO(H5E_RESOURCE, H5E_OVERFLOW, 0, "uncompressed buffer overrun");

        free(*buf);
        *buf = newbuf;
        newbuf = 0;
        *buf_size = zsize;
        retval = zsize;
    }

done:
    if (zfld) Z zfp_field_free(zfld);
    if (zstr) Z zfp_stream_close(zstr);
    if (bstr) B stream_close(bstr);
    if (newbuf) free(newbuf);
    return retval ;
}

#undef Z
#undef B
