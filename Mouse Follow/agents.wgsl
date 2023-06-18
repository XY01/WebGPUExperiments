//include:math.wgsl

// Pixels
@group(0) @binding(0)  
  var<storage, read_write> pixels : array<vec4f>;

// Uniforms
@group(1) @binding(0) 
  var<uniform> rez : f32;

@group(1) @binding(1) 
  var<uniform> time : f32;

@group(1) @binding(2) 
  var<uniform> mousePos : vec2f;

@group(1) @binding(3) 
  var<uniform> canvasScale : f32;

@group(1) @binding(4) 
  var<uniform> agentCount : u32;

// Agent positions
@group(2) @binding(0)  
  var<storage, read_write> positions : array<vec2f>;

// Agent velocities
@group(2) @binding(1)  
  var<storage, read_write> velocities : array<vec2f>;


////////////////////////////////////////
// custom maths
fn lerp(x: f32, y: f32, t: f32) -> f32
{
    return (1.0 - t) * x + t * y;
}

fn cross(a: vec2f, b: vec2f) -> f32
{
    return a.x * b.y - a.y * b.x;
}

fn rotate(v: vec2f, theta: f32 ) -> vec2f{
    var s = sin(theta);
    var c = cos(theta);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

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
  var y = r(f32(id.x) * 2.0);
  var p = vec2(x, y);
  p *= rez;
  positions[id.x] = p;

  velocities[id.x] = vec2(r(f32(id.x+1)), r(f32(id.x + 2))) - 0.5;
}

@compute @workgroup_size(256)
fn simulate(@builtin(global_invocation_id) id : vec3u) 
{
  var rand = r(f32(id.x));
  var v : vec2f = velocities[id.x];
  var p : vec2f = positions[id.x];

  //-- Normalizing positions to get direction
  // mouse pos is in canvas space but the positions are calculated in 
  // the canvas rez since its drawn to a texture of x rez
  var mouseNorm : vec2f = mousePos / canvasScale;
  var posNorm : vec2f = p/rez;
  var dirToMouse : vec2f = normalize(mouseNorm - posNorm); 


  // limit turn rate
  var currentDirection = normalize(v);
  var angleDiff = acos(dot(currentDirection, dirToMouse));
  var maxTurnRadians = lerp(0.2, 0.9, rand);
  if(angleDiff > maxTurnRadians)
  {
    var turnDir = sign(cross(currentDirection,dirToMouse));
    dirToMouse = rotate(currentDirection, maxTurnRadians * turnDir);
  }

  // Limit and vary steerForceing force
  var maxSpeed = 4.0 + rand;
  var desiredVel = dirToMouse * maxSpeed;
  var steerForce = desiredVel - v;
  var maxForce = lerp(0.1, 0.4, rand);
  steerForce = normalize(steerForce) * maxForce;

  // avoid force
  var count : u32 = 0;
  var avoidForce : vec2f = vec2f(0,0);
  var effectDist = rez * 0.3;
  for (var i : u32 = 0; i < agentCount; i++)
  {
    if(i != id.x)
    {
        var vecToPos = p - positions[i];
        var dist = length(vecToPos);        
        if(dist < effectDist)
        {
          var distScalar = pow((1-clamp(dist / effectDist,0,1)),3);
          avoidForce = normalize(vecToPos) * distScalar * 1;
        }
    }
  }

  // Acceleration and drag
  var acc = steerForce;
  acc += avoidForce;
  v += acc;

  var drag = -v * lerp(0.005, 0.1, 1-rand);
  v += drag;


  // Clamp velocities
  var clampSpeed = 8.0;
  v.x = clamp(v.x, -clampSpeed, clampSpeed);
  v.y = clamp(v.y, -clampSpeed, clampSpeed);


  // Accumulated
  p += v;

  // wrap around rez
  p = (p + rez) % rez;

  positions[id.x] = p;
  velocities[id.x] = v;

  var red : f32 = pow(length(v)/5,3);
  
  pixels[index(p)] = vec4(red, 0.0, 0.0, 1.0);
}


@compute @workgroup_size(256)
fn fade(@builtin(global_invocation_id) id : vec3u) {
  pixels[id.x] *= 0.90;
}