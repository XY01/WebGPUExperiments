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

@group(2) @binding(2)  
  var<storage, read_write> states : array<u32>;


fn r(n: f32) -> f32 {
  let x = sin(n) * 43758.5453;
  return fract(x);
}

fn index(p: vec2f) -> i32 {
  return i32(p.x) + i32(p.y) * i32(rez);
}

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) id : vec3u) {
  let seed = f32(id.x)/f32(count);
  var p = vec2(r(seed), r(seed + 0.1));
  p *= rez;
  positions[id.x] = p;

  velocities[id.x] = vec2(r(f32(id.x+1)), r(f32(id.x + 2))) - 0.5;
  velocities[id.x] *= 3.0;
  states[id.x] = 1;

  // Set static agent
  states[0] = 0;
   positions[0] = vec2(rez/2.0);
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id : vec3u) {
  var p = positions[id.x];
  var v = velocities[id.x];

  // If not moving then draw pixel and return
  if(states[id.x] == 0)
  {
    pixels[index(p)] = vec4(1.0, 0.0, 0.0, 1.0);
    return;
  }

  // search all other agents, check if static, then check distance
  // If too close then change state to static
  for(var i: u32 = 0; i < count; i++)
  {
    if(i!=id.x)
    {     
      if(states[i] == 0)
      {
        if(length(p-positions[i]) <= 1)
        {
          states[id.x] = 0;
        }
      }
    }
  }

  if(states[id.x] == 0)
  {
    v *= 0;
  }


  p += v;
  p = (p + rez) % rez;

  positions[id.x] = p;

  pixels[index(p)] = vec4(1.0, 1.0, 1.0, 1.0);
}


@compute @workgroup_size(256)
fn fade(@builtin(global_invocation_id) id : vec3u) {
  pixels[id.x] *= 0.90;
}