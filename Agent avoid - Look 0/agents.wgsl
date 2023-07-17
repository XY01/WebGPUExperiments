// Pixels
@group(0) @binding(0)  
  var<storage, read_write> pixels : array<vec4f>;

// Uniforms
@group(1) @binding(0) 
  var<uniform> rez : f32;

@group(1) @binding(1) 
  var<uniform> time : f32;

@group(1) @binding(2) 
  var<uniform> count : u32;

// Agent positions
@group(2) @binding(0)  
  var<storage, read_write> positions : array<vec2f>;

// Agent velocities
@group(2) @binding(1)  
  var<storage, read_write> velocities : array<vec2f>;


fn r(n: f32) -> f32 {
  let x = sin(n) * 43758.5453;
  return fract(x);
}

fn index(p: vec2f) -> i32 {
  return i32(p.x) + i32(p.y) * i32(rez);
}

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id : vec3u) {
  var x = r(f32(id.x));
  var y = r(f32(id.x) * 2);
  var p = vec2(x, y);
  p *= rez;
  positions[id.x] = p;

  velocities[id.x] = vec2(r(f32(id.x+1)), r(f32(id.x + 2))) - 0.5;
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id : vec3u) {
  var p = positions[id.x];
  var v = velocities[id.x];

  // Update position
  p += v;
  p = (p + rez) % rez;
  positions[id.x] = p;

  // Update velocity
  var avoidV = vec2(0.0);
  for (var i = 0u; i < count; i++) {
    if (i == id.x) { continue; }
    var other = positions[i]%400;
    var d = distance(other, p);
    if (d < 10) {
      avoidV += p - other;
    }
  }
  v += 0.05 * avoidV;
  v = normalize(v);
  velocities[id.x] = v;

  pixels[index(p)] = vec4(0.0, 1.0, 0.0, 1.0);
}


@compute @workgroup_size(256)
fn fade(@builtin(global_invocation_id) id : vec3u) {
  pixels[id.x] *= 0.90;
}