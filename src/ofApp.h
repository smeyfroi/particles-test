#pragma once

#include "ofMain.h"
#include <vector>

// Particle structure matching shader layout
struct Particle {
    ofVec2f position;
    ofVec2f velocity;
};

// Particle with Morton code for spatial sorting
struct ParticleWithMorton {
    int index;
    uint32_t mortonCode;
    ofVec2f position;
};

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);
    
    // Particle count
    static constexpr int NUM_PARTICLES = 2000;
    
private:
    void setupShaders();
    void setupParticles();
    void updatePhysics();
    void spatialSort();
    uint32_t calculateMortonCode(float x, float y);
    uint32_t expandBits(uint32_t v);
    
    // Physics parameters
    float timeStep;
    float damping;
    float attractionStrength;
    float repulsionStrength;
    float maxSpeed;
    
    // Rendering parameters
    float connectionDistance;
    float particleSize;
    
    // Visualization modes
    bool showSortedConnections;
    bool drawLines;
    int adaptiveSearchRadius;
    
    // Performance profiling
    float avgPhysicsTime;
    float avgSortTime;
    float avgRenderTime;
    int frameCounter;
    
    // Shaders
    ofShader particleUpdateShader;
    ofShader particleRenderShader;
    ofShader lineRenderShader;
    
    // Transform feedback VAOs and VBOs (ping-pong buffers)
    GLuint vao[2];
    GLuint vbo[2];
    int currentBuffer;
    
    // Line rendering with geometry shader
    ofVbo particleProxyVbo;
    std::vector<ofVec2f> particlePositionsOnly;
    
    // Texture buffer for particle positions (accessible in geometry shader)
    GLuint particlePositionTBO;  // Texture Buffer Object
    GLuint particlePositionTexture;  // Texture handle
    
    // Spatial sorting using Morton codes
    std::vector<ParticleWithMorton> particlesWithMorton;
    std::vector<int> sortedIndices;
    GLuint sortedIndicesTBO;
    GLuint sortedIndicesTexture;
    bool useSpatialSort;
    
    std::vector<Particle> particles;
    
    // Mouse interaction
    bool attractMode;
};
