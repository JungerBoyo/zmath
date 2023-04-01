pub const Vec2u32 = Vec2(u32); pub const Vec2u64 = Vec2(u64);
pub const Vec3u32 = Vec3(u32); pub const Vec3u64 = Vec3(u64);
pub const Vec4u32 = Vec4(u32); pub const Vec4u64 = Vec4(u64);

pub const Vec2i32 = Vec2(i32); pub const Vec2i64 = Vec2(i64);
pub const Vec3i32 = Vec3(i32); pub const Vec3i64 = Vec3(i64);
pub const Vec4i32 = Vec4(i32); pub const Vec4i64 = Vec4(i64);

pub const Vec2f32 = Vec2(f32); pub const Vec2f64 = Vec2(f64);
pub const Vec3f32 = Vec3(f32); pub const Vec3f64 = Vec3(f64);
pub const Vec4f32 = Vec4(f32); pub const Vec4f64 = Vec4(f64);

pub fn Vec2(comptime T: type) type { return Vec(T, 2); }
pub fn Vec3(comptime T: type) type { return Vec(T, 3); }
pub fn Vec4(comptime T: type) type { return Vec(T, 4); }

pub const Mat2u32 = Mat2(u32); pub const Mat2u64 = Mat2(u64);
pub const Mat3u32 = Mat3(u32); pub const Mat3u64 = Mat3(u64);
pub const Mat4u32 = Mat4(u32); pub const Mat4u64 = Mat4(u64);

pub const Mat2i32 = Mat2(i32); pub const Mat2i64 = Mat2(i64);
pub const Mat3i32 = Mat3(i32); pub const Mat3i64 = Mat3(i64);
pub const Mat4i32 = Mat4(i32); pub const Mat4i64 = Mat4(i64);

pub const Mat2f32 = Mat2(f32); pub const Mat2f64 = Mat2(f64);
pub const Mat3f32 = Mat3(f32); pub const Mat3f64 = Mat3(f64);
pub const Mat4f32 = Mat4(f32); pub const Mat4f64 = Mat4(f64);

pub fn Mat2(comptime T: type) type { return Mat(Vec2(T), 2); }
pub fn Mat3(comptime T: type) type { return Mat(Vec3(T), 3); }
pub fn Mat4(comptime T: type) type { return Mat(Vec4(T), 4); }

pub const Quaternionf32 = Quaternion(f32);
pub const Quaternionf64 = Quaternion(f64);

const PI = @import("std").math.pi;

pub fn Vec(comptime T: type, comptime size: comptime_int) type {
  if (@typeInfo(T) != .Float and @typeInfo(T) != .Int) {
    @compileError("Vector underlying type must be floating point or integer");
  }

  if (size < 2 or size > 4) {
    @compileError("Possible vectors are Vec2, Vec3 and Vec4.");
  }

  return extern struct {
    const Self = @This();

    values: [size]T,

    pub fn UnderlyingType() type {
      return T;
    }

    pub fn Size() comptime_int {
      return size;
    }

    pub fn len(self: Self) T {
      return @sqrt(Self.dot(self, self));
    }

    pub fn dot(lhs: Self, rhs: Self) T {
      var sum: T = 0;

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        sum += lhs.values[i] * rhs.values[i];
      }

      return sum;
    }

    pub fn normalize(self: Self) Self {
      return self.divScalar(self.len());
    }

    pub fn negate(self: Self) Self {
      var result: Self = undefined;

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result.values[i] = -self.values[i];
      }

      return result;
    }

    pub fn sub(lhs: Self, rhs: Self) Self {
      var result: Self = undefined;

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result.values[i] = lhs.values[i] - rhs.values[i];
      }

      return result;
    }

    pub fn add(lhs: Self, rhs: Self) Self {
      var result: Self = undefined;

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result.values[i] = lhs.values[i] + rhs.values[i];
      }

      return result;
    }

    pub fn mul(lhs: Self, rhs: Self) Self {
      var result: Self = undefined;
      
      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result.values[i] = lhs.values[i] * rhs.values[i];
      }

      return result;
    }

    pub fn mulScalar(vec: Self, scalar: T) Self {
      var result: Self = undefined;

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result.values[i] = vec.values[i] * scalar;
      }

      return result;
    }

    pub fn divScalar(vec: Self, scalar: T) Self {
      var result: Self = undefined;

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result.values[i] = vec.values[i] / scalar;
      }

      return result;
    }

    pub fn accumulate(vecs: []const Self) Self {
      var result = Self.initAll(0);

      for (vecs) |vec| {
        result = result.add(vec);
      }

      return result;
    }

    pub fn x(self: Self) T {
      return self.values[0];
    }

    pub fn y(self: Self) T {
      return self.values[1];
    }

    pub usingnamespace switch (size) {
      2 => extern struct {
        pub fn initAll(value: T) Self {
          return .{ .values = .{value, value}};
        }

        pub fn init(vx: T, vy: T) Self {
          return .{ .values = .{vx, vy} };
        } 
      },
      3 => extern struct {
        pub fn initAll(value: T) Self {
          return .{ .values = .{value, value, value}};
        }

        pub fn init(vx: T, vy: T, vz: T) Self {
          return .{ .values = .{vx, vy, vz} };
        }

        pub fn z(self: Self) T {
          return self.values[2];
        }

        pub fn cross(lhs: Self, rhs: Self) Self {
          return .{ .values = .{ 
            lhs.values[1]*rhs.values[2] - rhs.values[1]*lhs.values[2], 
            lhs.values[2]*rhs.values[0] - rhs.values[2]*lhs.values[0],
            lhs.values[0]*rhs.values[1] - rhs.values[0]*lhs.values[1], 
          }};
        }
      },
      4 => extern struct {
        pub fn initAll(value: T) Self {
          return .{ .values = .{value, value, value, value}};
        }

        pub fn z(self: Self) T {
          return self.values[2];
        }
        
        pub fn w(self: Self) T {
          return self.values[3];
        }

        pub fn init(vx: T, vy: T, vz: T, vw: T) Self {
          return .{ .values = .{vx, vy, vz, vw} };
        }
      },
      else => {}
    };
  };
}

pub fn Mat(comptime T: type, comptime size: comptime_int) type {
  if (T.Size() != size) {
    @compileError("Dimensions of the matrix must be equal");
  }

  if (T != Vec(f32, size) and T != Vec(f64, size)) {
    @compileError("Mat can be composed only from Vec(T, size) type");
  }

  return extern struct { 
    const Self = @This();

    values: [size]T,

    pub fn mulVec(lhs: Self, rhs: T) T {
      var result = T.initAll(0);

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result = result.add(lhs.values[i].mulScalar(rhs.values[i]));
      }

      return result;
    }

    pub fn mulScalar(lhs: Self, rhs: T.UnderlyingType()) Self {
      var result: Self = undefined;

      comptime var i = 0;
      inline while (i < size) : (i += 1) {
        result.values[i] = lhs.values[i].mulScalar(rhs);
      }

      return result;
    }

    pub fn initFromVec(vecs: [size]T) Self {
      return .{ .values = vecs };
    }

    pub fn at(self: Self, c: usize, r: usize) T.UnderlyingType() {
      return self.values[c].values[r];
    }

    pub usingnamespace switch(size) {
      2 => extern struct {
        pub fn init() Self {
          return .{ .values = .{
            T.initAll(0),
            T.initAll(0),
          }};
        }
        pub fn initDiagonal(value: T.UnderlyingType()) Self {
          return .{ .values = .{
            T.init(value, 0),
            T.init(0, value),
          }};
        }
        pub fn transpose(self: Self) Self {
          return .{ .values = .{
            .{
              self.values[0].values[0], 
              self.values[1].values[0], 
            },
            .{
              self.values[0].values[1], 
              self.values[1].values[1], 
            },
          }};
        }
        pub fn mul(lhs: Self, rhs: Self) Self {
          return .{ .values = .{
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[0].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[0].values[1]),
            }),
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[1].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[1].values[1]),
            }),
          }};
        }

        pub fn det(self: Self) T.UnderlyingType() {
          return self.values[0].values[0]*self.values[1].values[1] - 
                 self.values[0].values[1]*self.values[1].values[0];
        }

        pub fn inverse(self: Self) Self {
          const d = @as(T.UnderlyingType(), 1)/self.det();
          return .{ .values = .{
            .{ d *  self.values[1].values[1], d * -self.values[1].values[0] },
            .{ d * -self.values[0].values[1], d *  self.values[0].values[0] },
          }};
        }
      },
      3 => extern struct {
        pub fn init() Self {
          return .{ .values = .{
            T.initAll(0),
            T.initAll(0),
            T.initAll(0),
          }};
        }
        pub fn initDiagonal(value: T.UnderlyingType()) Self {
          return .{ .values = .{
            T.init(value, 0, 0),
            T.init(0, value, 0),
            T.init(0, 0, value),
          }};
        }
        pub fn transpose(self: Self) Self {
          return .{ .values = .{
            .{
              self.values[0].values[0], 
              self.values[1].values[0], 
              self.values[2].values[0], 
            },
            .{
              self.values[0].values[1], 
              self.values[1].values[1], 
              self.values[2].values[1], 
            },
            .{
              self.values[0].values[2], 
              self.values[1].values[2], 
              self.values[2].values[2], 
            },
          }};
        }
        pub fn mul(lhs: Self, rhs: Self) Self {
          return .{ .values = .{
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[0].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[0].values[1]),
              T.mulScalar(lhs.values[2], rhs.values[0].values[2]),
            }),
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[1].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[1].values[1]),
              T.mulScalar(lhs.values[2], rhs.values[1].values[2]),
            }),
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[2].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[2].values[1]),
              T.mulScalar(lhs.values[2], rhs.values[2].values[2]),
            }),
          }};
        }
        pub fn det(self: Self) T.UnderlyingType() {
          return 
            self.at(0, 0) * (self.at(1, 1) * self.at(2, 2) - self.at(2, 1) * self.at(1, 2)) -
            self.at(0, 1) * (self.at(1, 0) * self.at(2, 2) - self.at(1, 2) * self.at(2, 0)) + 
            self.at(0, 2) * (self.at(1, 0) * self.at(2, 1) - self.at(1, 1) * self.at(2, 0))
          ;
        }
        pub fn inverse(self: Self) Self {
          const d = @as(T.UnderlyingType(), 1) / self.det();
          return .{ .values = .{
            (self.at(1, 1) * self.at(2, 2) - self.at(2, 1) * self.at(1, 2)) * d,
            (self.at(0, 2) * self.at(2, 1) - self.at(0, 1) * self.at(2, 2)) * d,
            (self.at(0, 1) * self.at(1, 2) - self.at(0, 2) * self.at(1, 1)) * d,
            (self.at(1, 2) * self.at(2, 0) - self.at(1, 0) * self.at(2, 2)) * d,
            (self.at(0, 0) * self.at(2, 2) - self.at(0, 2) * self.at(2, 0)) * d,
            (self.at(1, 0) * self.at(0, 2) - self.at(0, 0) * self.at(1, 2)) * d,
            (self.at(1, 0) * self.at(2, 1) - self.at(2, 0) * self.at(1, 1)) * d,
            (self.at(2, 0) * self.at(0, 1) - self.at(0, 0) * self.at(2, 1)) * d,
            (self.at(0, 0) * self.at(1, 1) - self.at(1, 0) * self.at(0, 1)) * d,
          }};
        }
      },
      4 => extern struct {
        pub fn init() Self {
          return .{ .values = .{
            T.initAll(0),
            T.initAll(0),
            T.initAll(0),
            T.initAll(0),
          }};
        }
        pub fn initDiagonal(value: T.UnderlyingType()) Self {
          return .{ .values = .{
            T.init(value, 0, 0, 0),
            T.init(0, value, 0, 0),
            T.init(0, 0, value, 0),
            T.init(0, 0, 0, value),
          }};
        }
        pub fn transpose(self: Self) Self {
          return .{ .values = .{
            .{
              self.values[0].values[0], 
              self.values[1].values[0], 
              self.values[2].values[0], 
              self.values[3].values[0], 
            },
            .{
              self.values[0].values[1], 
              self.values[1].values[1], 
              self.values[2].values[1], 
              self.values[3].values[1], 
            },
            .{
              self.values[0].values[2], 
              self.values[1].values[2], 
              self.values[2].values[2], 
              self.values[3].values[2], 
            },
            .{
              self.values[0].values[3], 
              self.values[1].values[3], 
              self.values[2].values[3], 
              self.values[3].values[3], 
            },
          }};
        }
        pub fn mul(lhs: Self, rhs: Self) Self {
          return .{ .values = .{
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[0].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[0].values[1]),
              T.mulScalar(lhs.values[2], rhs.values[0].values[2]),
              T.mulScalar(lhs.values[3], rhs.values[0].values[3]),
            }),
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[1].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[1].values[1]),
              T.mulScalar(lhs.values[2], rhs.values[1].values[2]),
              T.mulScalar(lhs.values[3], rhs.values[1].values[3]),
            }),
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[2].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[2].values[1]),
              T.mulScalar(lhs.values[2], rhs.values[2].values[2]),
              T.mulScalar(lhs.values[3], rhs.values[2].values[3]),
            }),
            T.accumulate(&[_]T{
              T.mulScalar(lhs.values[0], rhs.values[3].values[0]),
              T.mulScalar(lhs.values[1], rhs.values[3].values[1]),
              T.mulScalar(lhs.values[2], rhs.values[3].values[2]),
              T.mulScalar(lhs.values[3], rhs.values[3].values[3]),
            }),
          }};
        }
        pub fn initAsRotationFromUnitQuaternion(q: Quaternion(T.UnderlyingType())) Self {
          return .{ .values = .{
            T.init(
              1.0 - 2.0*(q.j*q.j + q.k*q.k), 
              2.0*(q.i*q.j + q.w*q.k), 
              2.0*(q.i*q.k - q.w*q.j), 
              0.0
            ),
            T.init(
              2.0*(q.i*q.j - q.w*q.k), 
              1.0 - 2.0*(q.i*q.i + q.k*q.k), 
              2.0*(q.j*q.k + q.w*q.i), 
              0.0
            ),
            T.init(
              2.0*(q.i*q.k + q.w*q.j),
              2.0*(q.j*q.k - q.w*q.i),
              1.0 - 2.0*(q.i*q.i + q.j*q.j),
              0.0
            ),
            T.init(0.0, 0.0, 0.0, 1.0),
          }};
        }

        pub fn initAsRotationFromEulerAngles(angles: Vec3(T.UnderlyingType())) Self {
          const rot_x = if (angles.values[0] == 0.0) 
              initDiagonal(1.0) 
            else 
              initAsRotationFromUnitQuaternion(Quaternion(T.UnderlyingType()).init(.{ .values = .{1.0, 0.0, 0.0}}, angles.values[0]/2.0));

          const rot_y = if (angles.values[1] == 0.0) 
              initDiagonal(1.0) 
            else 
              initAsRotationFromUnitQuaternion(Quaternion(T.UnderlyingType()).init(.{ .values = .{0.0, 1.0, 0.0}}, angles.values[1]/2.0));
          
          const rot_z = if (angles.values[2] == 0.0) 
              initDiagonal(1.0) 
            else 
              initAsRotationFromUnitQuaternion(Quaternion(T.UnderlyingType()).init(.{ .values = .{0.0, 0.0, 1.0}}, angles.values[2]/2.0));

          return rot_z.mul(rot_y.mul(rot_x));
        }
        pub fn initAsSymmetricPerspectiveProjection(fov: f32, near: f32, far: f32, win_w: f32, win_h: f32) Self {
          return .{ .values = .{
            T.init(1.0/((win_w/win_h) * fov), 0.0, 0.0, 0.0),
            T.init(0.0, 1.0/fov, 0.0, 0.0),
            T.init(0.0, 0.0, (-far - near)/(far - near), -1.0),
            T.init(0.0, 0.0, -(2.0 * far * near)/(far - near), 0.0),
          }};
        }
        pub fn det(self: Self) Self {
          const A2323 = self.at(2, 2) * self.at(3, 3) - self.at(2, 3) * self.at(3, 2);
          const A1323 = self.at(2, 1) * self.at(3, 3) - self.at(2, 3) * self.at(3, 1);
          const A1223 = self.at(2, 1) * self.at(3, 2) - self.at(2, 2) * self.at(3, 1);
          const A0323 = self.at(2, 0) * self.at(3, 3) - self.at(2, 3) * self.at(3, 0);
          const A0223 = self.at(2, 0) * self.at(3, 2) - self.at(2, 2) * self.at(3, 0);
          const A0123 = self.at(2, 0) * self.at(3, 1) - self.at(2, 1) * self.at(3, 0);
          return 
            self.at(0, 0) * ( self.at(1, 1) * A2323 - self.at(1, 2) * A1323 + self.at(1, 3) * A1223 ) -
            self.at(0, 1) * ( self.at(1, 0) * A2323 - self.at(1, 2) * A0323 + self.at(1, 3) * A0223 ) +
            self.at(0, 2) * ( self.at(1, 0) * A1323 - self.at(1, 1) * A0323 + self.at(1, 3) * A0123 ) -
            self.at(0, 3) * ( self.at(1, 0) * A1223 - self.at(1, 1) * A0223 + self.at(1, 2) * A0123 )
          ;
        }
        pub fn inverse(self: Self) Self {
          const A2323 = self.at(2, 2) * self.at(3, 3) - self.at(2, 3) * self.at(3, 2);
          const A1323 = self.at(2, 1) * self.at(3, 3) - self.at(2, 3) * self.at(3, 1);
          const A1223 = self.at(2, 1) * self.at(3, 2) - self.at(2, 2) * self.at(3, 1);
          const A0323 = self.at(2, 0) * self.at(3, 3) - self.at(2, 3) * self.at(3, 0);
          const A0223 = self.at(2, 0) * self.at(3, 2) - self.at(2, 2) * self.at(3, 0);
          const A0123 = self.at(2, 0) * self.at(3, 1) - self.at(2, 1) * self.at(3, 0);
          const A2313 = self.at(1, 2) * self.at(3, 3) - self.at(1, 3) * self.at(3, 2);
          const A1313 = self.at(1, 1) * self.at(3, 3) - self.at(1, 3) * self.at(3, 1);
          const A1213 = self.at(1, 1) * self.at(3, 2) - self.at(1, 2) * self.at(3, 1);
          const A2312 = self.at(1, 2) * self.at(2, 3) - self.at(1, 3) * self.at(2, 2);
          const A1312 = self.at(1, 1) * self.at(2, 3) - self.at(1, 3) * self.at(2, 1);
          const A1212 = self.at(1, 1) * self.at(2, 2) - self.at(1, 2) * self.at(2, 1);
          const A0313 = self.at(1, 0) * self.at(3, 3) - self.at(1, 3) * self.at(3, 0);
          const A0213 = self.at(1, 0) * self.at(3, 2) - self.at(1, 2) * self.at(3, 0);
          const A0312 = self.at(1, 0) * self.at(2, 3) - self.at(1, 3) * self.at(2, 0);
          const A0212 = self.at(1, 0) * self.at(2, 2) - self.at(1, 2) * self.at(2, 0);
          const A0113 = self.at(1, 0) * self.at(3, 1) - self.at(1, 1) * self.at(3, 0);
          const A0112 = self.at(1, 0) * self.at(2, 1) - self.at(1, 1) * self.at(2, 0);

          const d = @as(T.UnderlyingType(), 1)/self.det();
          return .{ .values = .{
            .{
              d *   ( self.at(1, 1) * A2323 - self.at(1, 2) * A1323 + self.at(1, 3) * A1223 ),
              d * - ( self.at(0, 1) * A2323 - self.at(0, 2) * A1323 + self.at(0, 3) * A1223 ),
              d *   ( self.at(0, 1) * A2313 - self.at(0, 2) * A1313 + self.at(0, 3) * A1213 ),
              d * - ( self.at(0, 1) * A2312 - self.at(0, 2) * A1312 + self.at(0, 3) * A1212 ),
            },
            .{
              d * - ( self.at(1, 0) * A2323 - self.at(1, 2) * A0323 + self.at(1, 3) * A0223 ),
              d *   ( self.at(0, 0) * A2323 - self.at(0, 2) * A0323 + self.at(0, 3) * A0223 ),
              d * - ( self.at(0, 0) * A2313 - self.at(0, 2) * A0313 + self.at(0, 3) * A0213 ),
              d *   ( self.at(0, 0) * A2312 - self.at(0, 2) * A0312 + self.at(0, 3) * A0212 ),
            },
            .{
              d *   ( self.at(1, 0) * A1323 - self.at(1, 1) * A0323 + self.at(1, 3) * A0123 ),
              d * - ( self.at(0, 0) * A1323 - self.at(0, 1) * A0323 + self.at(0, 3) * A0123 ),
              d *   ( self.at(0, 0) * A1313 - self.at(0, 1) * A0313 + self.at(0, 3) * A0113 ),
              d * - ( self.at(0, 0) * A1312 - self.at(0, 1) * A0312 + self.at(0, 3) * A0112 ),
            },
            .{
              d * - ( self.at(1, 0) * A1223 - self.at(1, 1) * A0223 + self.at(1, 2) * A0123 ),
              d *   ( self.at(0, 0) * A1223 - self.at(0, 1) * A0223 + self.at(0, 2) * A0123 ),
              d * - ( self.at(0, 0) * A1213 - self.at(0, 1) * A0213 + self.at(0, 2) * A0113 ),
              d *   ( self.at(0, 0) * A1212 - self.at(0, 1) * A0212 + self.at(0, 2) * A0112 ),
            },
          }};
        }
      },
      else => {}
    };
  };
}

pub fn Quaternion(comptime T: type) type {
  if (@typeInfo(T) != .Float) {
    @compileError("Quaternion underlying type must be floating point");
  }

  return extern struct {
    const Self = @This();

    w: T = 0.0, i: T = 0.0, j: T = 0.0, k: T = 0.0,

    pub fn init(rotation_axis: Vec3(T), angle: T) Self {
      const sin = @sin(angle);
      return .{
        .w = @cos(angle),
        .i = sin * rotation_axis.values[0],
        .j = sin * rotation_axis.values[1],
        .k = sin * rotation_axis.values[2],
      };
    }
    
    pub fn initFromVec3(vec: Vec3(T)) Self {
      return .{ .i = vec.values[0], .j = vec.values[1], .k = vec.values[2] };
    }

    pub fn mul(self: Self, other: Self) Self {
      return .{
        .w = self.w * other.w - self.i * other.i - self.j * other.j - self.k * other.k,
        .i = self.w * other.i + self.i * other.w - self.k * other.j + self.j * other.k,
        .j = self.w * other.j + self.j * other.w - self.i * other.k + self.k * other.i,
        .k = self.w * other.k + self.k * other.w - self.j * other.i + self.i * other.j,
      };
    }

    pub fn negateIm(self: Self) Self {
      return .{ .w = self.w, .i = -self.i, .j = -self.j, .k = -self.k };
    }
  };
}

pub fn transforms(comptime T: type) type {
  if (@typeInfo(T) != .Float) {
    @compileError("transforms underlying type must be floating point");
  }

  return extern struct {
    pub fn rotatePoint3d(angles: Vec3(T), point: Vec3(T), origin: Vec3(T), basis: Mat3(T)) Vec3(T) {
      const q_point = Quaternion(T).initFromVec3(point.sub(origin));

      const q1 = Quaternion(T).init(basis.mulVec(Vec3(T).init(1.0, 0.0, 0.0)), angles.values[0]/2.0);
      const q2 = Quaternion(T).init(basis.mulVec(Vec3(T).init(0.0, 1.0, 0.0)), angles.values[1]/2.0);
      const q3 = Quaternion(T).init(basis.mulVec(Vec3(T).init(0.0, 0.0, 1.0)), angles.values[2]/2.0);

      const rotated_q_point = 
        q3.mul(
          q2.mul(
            q1.mul(
              q_point
            ).mul(q1.negateIm())
          ).mul(q2.negateIm())
        ).mul(q3.negateIm());
        
      return .{ .values = .{
        rotated_q_point.i + origin.values[0],
        rotated_q_point.j + origin.values[1], 
        rotated_q_point.k + origin.values[2], 
      }};
    }
  };
}