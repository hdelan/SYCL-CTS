/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Copyright:	(c) 2017 by Codeplay Software LTD. All Rights Reserved.
//
*******************************************************************************/

#include "../common/common.h"

#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
#include "../../util/test_base_cuda.h"
#include <cuda.h>
#endif

#define TEST_NAME cuda_interop_constructors

namespace cuda_interop_constructors__ {
using namespace sycl_cts;


/** tests the constructors for CUDA inter-op
 */
class TEST_NAME :
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    public util::test_base_cuda
#else
    public util::test_base
#endif
{
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute this test
   */
  void run(util::logger &log) override {
#ifdef SYCL_EXT_ONEAPI_BACKEND_CUDA
    {
      auto queue = util::get_cts_object::queue();
      if (queue.get_backend() != sycl::backend::ext_oneapi_cuda) {
        WARN(
            "CUDA interoperability part is not supported on non-CUDA "
            "backend types");
        return;
      }

      auto res = cuDeviceGet(&m_cu_device, 0);
      m_cu_platform.push_back(m_cu_device);

      /** check make_platform (std::vector<CUdevice>)
       */
      {
        sycl::platform platform =
            sycl::make_platform<sycl::backend::ext_oneapi_cuda>(&m_cu_platform);

	std::vector<CUdevice>* interopPlatformID =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(platform);
        if (*interopPlatformID != m_cu_platform) {
          FAIL(log, "platform was not constructed correctly");
        }
      }

      /** check make_device (CUdevice)
       */
      {
        sycl::device device =
            sycl::make_device<sycl::backend::ext_oneapi_cuda>(m_cu_device);

        CUdevice interopDeviceID =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
        if (interopDeviceID != m_cu_device) {
          FAIL(log, "device was not constructed correctly");
        }
      }
    }
#else
    log.note("The test is skipped because CUDA back-end is not supported");
#endif  // SYCL_EXT_ONEAPI_BACKEND_CUDA
  }
};

// register this test with the test_collection
util::test_proxy<TEST_NAME> proxy;

} /* namespace cuda_interop_constructors__ */
