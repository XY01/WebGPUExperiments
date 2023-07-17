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


// Other buffers
@group(2) @binding(0)  
  var<storage, read_write> positions : array<vec2f>;

@group(2) @binding(1)  
  var<storage, read_write> velocities : array<vec2f>;


fn r(n: f32) -> f32 {
  let x = sin(n) * 43758.5453;
  return fract(x);
}

fn index(p: vec2f) -> i32 {
  return i32(p.x) + i32(p.y) * i32(rez);
}


///////////////////////////////////
// FBM - From book of shaders

fn random (st : vec2f) -> f32 {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

// Based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
fn noise(st : vec2f) -> f32 {
    let i = floor(st);
    let f = fract(st);

    // Four corners in 2D of a tile
    let a : f32 = random(i);
    let b : f32 = random(i + vec2(1.0, 0.0));
    let c : f32 = random(i + vec2(0.0, 1.0));
    let d : f32 = random(i + vec2(1.0, 1.0));

    let u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

fn fbm(st : vec2f) -> f32 {
    // Initial values
    var value : f32 = 0.0;
    var amplitude : f32 = 0.5;
    var octaves : u32 = 8;
    var newSt : vec2f = st; // WORKAROUND: Had to create a new ST for some reason I can't change the input st value in wgsl

    //
    // Loop of octaves
    for (var i : u32 = 0; i < octaves; i++) {
        value += amplitude * noise(newSt);
        newSt *= 1.0;        // scale freq
        amplitude *= 0.5;  // scale amp
    }
    return value;
}

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id : vec3u) {
  let seed = f32(id.x)/f32(count);
  var p = vec2(r(seed), r(seed + 0.1));
  p *= rez;
  positions[id.x] = p;

  var fbmAccX = fbm(vec2f(p.xy)) - 0.5;
  var fbmAccY = fbm(vec2f(p.yx)) - 0.5;
  var vel : vec2f = vec2f(fbmAccX, fbmAccY) * 2.0;

  velocities[id.x] = vel;
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id : vec3u) {
  var p = positions[id.x];
  var v = velocities[id.x];

  // FBM Acc
  var sinTime = sin(time) * 20;
  var fbmAccX = fbm(vec2f(p.xy + sinTime)) - 0.5;
  var fbmAccY = fbm(vec2f(p.yx + sinTime)) - 0.5;
  var fbmAcc : vec2f = vec2f(fbmAccX, fbmAccY);
 
  // Collision
  // If within range, acc away at double impact speed
   var colAcc : vec2f = vec2f(0,0);
   for(var i=0; i < i32(count); i++) 
   {
        if(i == i32(id.x)) {
      continue;
    }
      let otherPos = positions[i];
      let d =  distance(otherPos, p);

      if(d < 1.5)
      {
          colAcc += (p - otherPos.xy) / d;
      }
   }

 
  // Add acc and limit to max speed
 // v += colAcc;
  v += fbmAcc;
  var maxSpeed = 0.75;
  var minSpeed = 0.25;
  var speed = length(v);
  if(speed > maxSpeed)  {
    v = maxSpeed * (v / speed);
  }

  // if(speed < minSpeed)
  // {
  //   v = minSpeed * (v / speed);
  // }
  velocities[id.x] = v;

  p += v;
  p = (p + rez) % rez;

  positions[id.x] = p;

  pixels[index(p)] = vec4(speed/maxSpeed, 0.0, 0.0, 1.0);
}


@compute @workgroup_size(256)
fn fade(@builtin(global_invocation_id) id : vec3u) {
  pixels[id.x] *= 0.90;
}