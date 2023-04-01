# zmath
simple linear algebra library written in zig.
## implements
  - Vec{2, 3, 4}x{{f, i, u}x{32, 64}}
  > dot product, cross product, {+, -, *, /} ops
  - Mat{2x2, 3x3, 4x4}x{{f, i, u}x{32, 64}}
  > determinant, inverse (without det==0 test), transpose, mul op
  - Mat4x4{f32, f64}
  > init from unit quaternion, init as rotation from euler angles, projection
  - Quaternion{f32, f64}
  > mul op, init from 3d vector and angle
  - transforms{f32, f64}
  > 3d point rotation
  
