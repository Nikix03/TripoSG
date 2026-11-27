Shader "AETHRYX/Toon"
{
    Properties
    {
        _MainTex ("Color base", 2D) = "white" {}
        _Color ("Tinte", Color) = (1,1,1,1)
        _Cutoff ("Corte de alfa", Range(0,1)) = 0
        _BandSoftness ("Suavidad de sombra", Range(.01,.25)) = .075
        _EmissionMap ("Mapa de emisión", 2D) = "white" {}
        [HDR] _EmissionColor ("Emisión", Color) = (0,0,0,0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            CGPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"
            sampler2D _MainTex;
            float4 _MainTex_ST;
            fixed4 _Color;
            float _Cutoff;
            float _BandSoftness;
            sampler2D _EmissionMap;
            half4 _EmissionColor;
            struct Input { float4 vertex : POSITION; float3 normal : NORMAL; float2 uv : TEXCOORD0; };
            struct Varying { float4 clip : SV_POSITION; float3 normal : TEXCOORD0; float2 uv : TEXCOORD1; };
            Varying Vert(Input input)
            {
                Varying output;
                output.clip = UnityObjectToClipPos(input.vertex);
                output.normal = UnityObjectToWorldNormal(input.normal);
                output.uv = TRANSFORM_TEX(input.uv, _MainTex);
                return output;
            }
            fixed4 Frag(Varying input) : SV_Target
            {
                fixed4 color = tex2D(_MainTex, input.uv) * _Color;
                clip(color.a - _Cutoff);
                float power = dot(normalize(input.normal), normalize(float3(-.32, .64, -.70)));
                float softness = max(_BandSoftness, fwidth(power));
                float shade = .64 + .19 * smoothstep(.13 - softness, .13 + softness, power)
                                   + .17 * smoothstep(.55 - softness, .55 + softness, power);
                half3 emission = tex2D(_EmissionMap, input.uv).rgb * _EmissionColor.rgb;
                return half4(color.rgb * shade + emission, color.a);
            }
            ENDCG
        }
    }
    Fallback "Standard"
}
