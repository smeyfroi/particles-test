AGENTS guide for this repository (openFrameworks C++ macOS)

Build/lint/test
- Build (Makefile):
  - Debug: make Debug
  - Release: make Release
  - Clean: make clean
  - Run (from bin): ./bin/particles-test.app/Contents/MacOS/particles-test
- Build (Xcode): open particles-test.xcodeproj and use the "particles-test Debug/Release" schemes
- Lint/format: use clang-format (LLVM style, C++17-friendly). Check only: clang-format -n -Werror $(git ls-files "*.{h,hpp,cpp,cxx}") ; Fix: clang-format -i ...
- Tests: none present. If adding tests, prefer Catch2 or GoogleTest; single test example (gtest): ctest -R NameRegex or bazel/meson as appropriate.

Code style
- Imports/includes: use angle brackets for OF/system headers (<ofMain.h>), quotes for local headers ("ofApp.h"); group and order: C++ std, third-party, openFrameworks, local.
- Formatting: 2 spaces indent; braces on same line for functions/classes; max line length ~100; run clang-format before committing.
- Types: prefer explicit fixed-width types (std::int32_t, std::size_t); use const & for non-owning; use auto only when type is obvious.
- Naming: PascalCase for types/classes, camelCase for functions/variables, UPPER_SNAKE_CASE for macros/const globals; member fields with trailing _ or mPrefix (pick one and stay consistent).
- Error handling: avoid exceptions in per-frame code; check return values; log via ofLogNotice/Warning/Error; validate pointers; guard GPU calls.
- Resource management: prefer RAII, std::unique_ptr/std::shared_ptr; avoid raw new/delete; use ofScopedLock for threads.
- Build config: set C++ standard in config.make if needed (uncomment MAC_OS_CPP_VER) and ensure minimum macOS target matches std.
- Performance: avoid allocations in update/draw; pre-reserve vectors; use const and references; consider ofFbo/ofVbo for batching.
- Platform: paths via ofToDataPath; keep code portable across Debug/Release.

Cursor/Copilot rules
- No .cursor rules or .github/copilot-instructions.md found in this project at generation time.
