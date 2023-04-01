const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.c
    const mode = b.standardReleaseOptions();

    const lib = b.addStaticLibrary("zmath", "src/math.zig");
    lib.setBuildMode(mode);
    lib.install();

    const main_tests = b.addTest("src/test.zig");
    main_tests.setBuildMode(mode);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.step);
}
