#pragma once

#include "ofMain.h"
#include <unordered_map>
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

// Spatial grid for fast proximity queries
class SpatialGrid {
public:
    float cellSize;
    std::unordered_map<int, std::vector<int>> grid;
    
    SpatialGrid() : cellSize(100.0f) {}  // Default constructor
    SpatialGrid(float size) : cellSize(size) {}
    
    void clear() { grid.clear(); }
    
    int hash(int x, int y) {
        return x * 73856093 ^ y * 19349663;
    }
    
    void insert(int particleIdx, const ofVec2f& pos) {
        int x = (int)floor(pos.x / cellSize);
        int y = (int)floor(pos.y / cellSize);
        grid[hash(x, y)].push_back(particleIdx);
    }
    
    void queryNeighbors(const ofVec2f& pos, float radius,
                       std::vector<int>& neighbors) {
        neighbors.clear();
        int minX = (int)floor((pos.x - radius) / cellSize);
        int maxX = (int)floor((pos.x + radius) / cellSize);
        int minY = (int)floor((pos.y - radius) / cellSize);
        int maxY = (int)floor((pos.y + radius) / cellSize);
        
        for (int x = minX; x <= maxX; x++) {
            for (int y = minY; y <= maxY; y++) {
                int h = hash(x, y);
                auto it = grid.find(h);
                if (it != grid.end()) {
                    neighbors.insert(neighbors.end(),
                                   it->second.begin(),
                                   it->second.end());
                }
            }
        }
    }
};

class ofApp : public ofBaseApp {
public:
    void setup();
    void update();
    void draw();
    void keyPressed(int key);
    
    // Particle count
    static constexpr int NUM_PARTICLES = 10000;
    
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
    bool showMortonColors;
    bool showSortedConnections;
    bool drawLines;  // Toggle line rendering
    bool useShaderRendering;  // Toggle between shader and CPU rendering
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
    
    // Spatial grid for comparison
    SpatialGrid spatialGrid;
    std::vector<Particle> particles;
    
    // Mouse interaction
    ofVec2f mouseForce;
    bool attractMode;
};
