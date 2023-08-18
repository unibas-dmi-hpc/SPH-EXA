#include <stdio.h>
#include "grackle_types.h"
#if !defined(GRACKLE_FLOAT_4) && !defined(GRACKLE_FLOAT_8)
#error "error"
#endif
grackle_version get_grackle_version() {
   	grackle_version out;
	out.version = "grackle_version";
	out.branch = "grackle_branch";
	out.revision = "grackle_revision";
	return out;
}
