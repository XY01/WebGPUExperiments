////////////////////////////////////////
// custom maths
fn lerp(x: f32, y: f32, t: f32) -> f32
{
    return (1.0 - t) * x + t * y;
}

fn cross2D(a: vec2f, b: vec2f) -> f32
{
    return a.x * b.y - a.y * b.x;
}

fn rotate2D(v: vec2f, theta: f32 ) -> vec2f{
    var s = sin(theta);
    var c = cos(theta);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}