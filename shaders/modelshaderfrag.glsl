#version 320 es

out lowp vec4 FragColor;
  
in lowp vec3 fPos;
in lowp vec2 fTexCoord;
in lowp vec3 fNormal;

//uniform sampler2D gPosition;
//uniform sampler2D gNormal;
//uniform sampler2D gAlbedo;
//uniform sampler2D ssao;

uniform lowp vec3 Ka;
uniform lowp vec3 Kd;
uniform lowp vec3 Ks;

struct Light {
    lowp vec3 Position;
    lowp vec3 Color;
    
    lowp float Linear;
    lowp float Quadratic;
    lowp float Radius;
};
Light light;

void main()
{        
    light.Position = vec3(5.0, 5.0, 5.0);
	light.Color = vec3(0.1, 0.1, 0.1);
	light.Linear = 0.009f;
	light.Quadratic = 0.0032f;
	    
    // retrieve data from gbuffer
    lowp vec3 FragPos = fPos;           //texture(fPos, fTexCoord).rgb;
    lowp vec3 Normal = fNormal;         //texture(fNormal, fTexCoord).rgb;
    lowp vec3 Diffuse = Kd;             //texture(gAlbedo, fTexCoord).rgb;
    lowp float AmbientOcclusion = 0.2; //texture(ssao, fTexCoord).r;
    
    // blinn-phong (in view-space)
    lowp vec3 ambient = vec3(0.3 * Diffuse * AmbientOcclusion); // here we add occlusion factor
    lowp vec3 lighting  = ambient; 
    lowp vec3 viewDir  = normalize(-FragPos); // viewpos is (0.0.0) in view-space
    // diffuse
    lowp vec3 lightDir = normalize(light.Position - FragPos);
    lowp vec3 diffuse = max(dot(Normal, lightDir), 0.0) * Diffuse * light.Color;
    // specular
    lowp vec3 halfwayDir = normalize(lightDir + viewDir);  
    lowp float spec = pow(max(dot(Normal, halfwayDir), 0.0), 8.0);
    lowp vec3 specular = light.Color * spec;
    // attenuation
    lowp float dist = length(light.Position - FragPos);
    lowp float attenuation = 1.0 / (1.0 + light.Linear * dist + light.Quadratic * dist * dist);
    diffuse  *= attenuation;
    specular *= attenuation;
    lighting += diffuse + specular;

    FragColor = vec4(Kd, 1.0);
}