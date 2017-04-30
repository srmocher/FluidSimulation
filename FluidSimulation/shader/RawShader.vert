float pointRadius=0.5;
float pointScale = 0.8;   // scale to calculate size in pixels
uniform mat4 current_projection_matrix;
uniform mat4 current_modelview_matrix;



varying vec3 posEye;        // position of center in eye space
layout(location = 0) in vec3 vert;
void main()
{

    posEye = vec3(current_modelview_matrix * vec4(vert, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = current_projection_matrix * vec4(vert, 1.0);

    gl_FrontColor = gl_Color;
}