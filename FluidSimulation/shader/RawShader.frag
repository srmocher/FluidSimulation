uniform float pointRadius;  // point size in world space
uniform float near;
uniform float far;
varying vec3 posEye;        // position of center in eye space

void main()
{
    // calculate normal from texture coordinates
    vec3 n;
    n.xy = gl_TexCoord[0].st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    //This is a more compatible version which works on ATI and Nvidia hardware
    //However, This does not work on Apple computers. :/
    //n.xy = gl_PointCoord.st*vec2(2.0, -2.0) + vec2(-1.0, 1.0);

    float mag = dot(n.xy, n.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
    n.z = sqrt(1.0-mag);

    // point on surface of sphere in eye space
    vec4 spherePosEye =vec4(posEye+n*pointRadius,1.0);

    vec4 clipSpacePos = gl_ProjectionMatrix*spherePosEye;
    float normDepth = clipSpacePos.z/clipSpacePos.w;

    // Transform into window coordinates coordinates
    gl_FragDepth = (((far-near)/2.)*normDepth)+((far+near)/2.);
    gl_FragData[0] = gl_Color;
}