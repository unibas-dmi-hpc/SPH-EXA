//
// Created by Noah Kubli on 28.11.22.
//

#ifndef SPHEXA_COOLER_H
#define SPHEXA_COOLER_H

extern "C" {
#include <grackle.h>
}
namespace cooling {

    struct GlobalValues {
        code_units units;
        chemistry_data data;
        chemistry_data_storage rates;
    };

    struct chemistry_data_
    {
        chemistry_data content;
    };
}
#endif //SPHEXA_COOLER_H
