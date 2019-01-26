#version 130

uniform mat4 projection_matrix;
uniform mat4 modelview_matrix;
uniform mat3 normal_matrix;

struct light {
	vec4 position;
	vec4 diffuse;
};

uniform light light0;


in vec3 a_Vertex;
in vec3 a_Normal;
//in vec2 a_TexCoord0;

out vec4 color;
//out vec2 texCoord0;

void main(void) 
{
	//texCoord0 = a_TexCoord0;
	
	vec4 pos = modelview_matrix * vec4(a_Vertex, 1.0);
	gl_Position = projection_matrix * pos;

	vec3 normalDir = normalize(normal_matrix * a_Normal);

	vec3 lightPos = light0.position.xyz;
	
	vec3 lightDir = normalize(lightPos - pos.xyz);

	float scale = 0.2+0.8*max(dot(lightDir, normalDir),0.0);
	//float scale = 0.9f;
	color = vec4(scale, scale, scale, 1.0);
}
