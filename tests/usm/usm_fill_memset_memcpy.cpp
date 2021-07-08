/*******************************************************************************
//
//  SYCL 2020 Conformance Test Suite
//
//  Checks functions fill, memset, and memcpy for Unified Shared Memory API.
//
*******************************************************************************/

#include "../common/common.h"
#include <cstring>

#define TEST_NAME usm_fill_memset_memcpy

namespace TEST_NAMESPACE {
using namespace sycl_cts;

/** Test instance
 */
class TEST_NAME : public sycl_cts::util::test_base {
 public:
  /** return information about this test
   */
  void get_info(test_base::info &out) const override {
    set_test_info(out, TOSTRING(TEST_NAME), TEST_FILE);
  }

  /** execute the test
   */
  void run(util::logger &log) override {
    try {
      auto q = util::get_cts_object::queue();
      if (!q.get_device().has(sycl::aspect::usm_shared_allocations)) {
        log.note(
            "Device does not support accessing to unified shared memory "
            "allocation");
        return;
      }

      constexpr auto count{2};
      constexpr auto size{sizeof(int) * count};
      constexpr auto value_for_filling{1};
      constexpr auto value_for_first_element_overwriting{10};

      auto deleter = [&q](int *data) { sycl::free(data, q.get_context()); };
      std::unique_ptr<int, decltype(deleter)> src(
          sycl::malloc_shared<int>(size, q.get_device(), q.get_context()),
          deleter);
      std::unique_ptr<int, decltype(deleter)> output(
          sycl::malloc_shared<int>(size, q.get_device(), q.get_context()),
          deleter);

      sycl::event init_fill = q.submit([&](sycl::handler &cgh) {
        cgh.fill(src.get(), value_for_filling, count);
      });
      sycl::event change_element = q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(init_fill);
        // overwrite the first element
        cgh.memset(src.get(), value_for_first_element_overwriting, sizeof(int));
      });
      q.submit([&](sycl::handler &cgh) {
        cgh.depends_on(change_element);
        cgh.memcpy(output.get(), src.get(), size);
      });
      q.wait_and_throw();
      // due to the peculiarities the sycl::handler::memset function, we perform
      // similar operations on an integer variable (expected_overwritten_value)
      int expected_overwritten_value{};
      std::memset(&expected_overwritten_value,
                  value_for_first_element_overwriting, sizeof(int));
      // check that the first element was overwritten correctly
      CHECK_VALUE_SCALAR(log, output.get()[0], expected_overwritten_value);
      for (int i = 1; i < count; i++) {
        CHECK_VALUE_SCALAR(log, output.get()[i], value_for_filling);
      }
    } catch (const sycl::exception &e) {
      log_exception(log, e);
      auto errorMsg = "a SYCL exception was caught: " + std::string(e.what());
      FAIL(log, errorMsg);
    }
  }
};

// construction of this proxy will register the above test
util::test_proxy<TEST_NAME> proxy;
}  // namespace TEST_NAMESPACE
