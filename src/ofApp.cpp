#include "ofApp.h"
#include <algorithm>
#include <cmath>
#include <chrono>

void ofApp::setup() {
    ofSetFrameRate(60);
    ofSetVerticalSync(true);
    ofBackground(0);
    
    cout << "Setting up particle system..." << endl;
    cout << "Window size: " << ofGetWidth() << "x" << ofGetHeight() << endl;
    
    // Initialize parameters
    timeStep = 0.016f;
    damping = 0.98f;
    attractionStrength = 50.0f;
    repulsionStrength = 100.0f;
    maxSpeed = 500.0f;
    connectionDistance = 80.0f;
    particleSize = 5.0f;  // Larger for easier visibility
    currentBuffer = 0;
    attractMode = true;
    useSpatialSort = true;  // Enable spatial sorting
    
    // Visualization
    showMortonColors = false;
    showSortedConnections = false;
    drawLines = false;  // Start with lines disabled to test particles
    useShaderRendering = false;  // Start with CPU rendering for debugging
    adaptiveSearchRadius = 500;  // Default search window
    
    // Performance tracking
    avgPhysicsTime = 0.0f;
    avgSortTime = 0.0f;
    avgRenderTime = 0.0f;
    frameCounter = 0;
    
    // Initialize spatial grid (cell size = connection distance for efficiency)
    spatialGrid = SpatialGrid(connectionDistance);
    
    cout << "Setting up shaders..." << endl;
    setupShaders();
    
    cout << "Setting up particles..." << endl;
    setupParticles();
    
    cout << "Setup complete! Press 'L' to toggle line rendering." << endl;
}

void ofApp::setupShaders() {
    // Particle update shader (Transform Feedback)
    string updateVert = R"(
        #version 410
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 velocity;
        
        uniform float deltaTime;
        uniform float damping;
        uniform vec2 mousePos;
        uniform float mouseStrength;
        uniform vec2 bounds;
        uniform float maxSpeed;
        
        out vec2 outPosition;
        out vec2 outVelocity;
        
        void main() {
            vec2 pos = position;
            vec2 vel = velocity;
            
            // Mouse force
            vec2 toMouse = mousePos - pos;
            float dist = length(toMouse);
            if (dist > 0.1 && dist < 300.0) {
                vec2 force = normalize(toMouse) * mouseStrength / (dist * 0.1);
                vel += force * deltaTime;
            }
            
            // Update position
            pos += vel * deltaTime;
            
            // Boundary bouncing
            if (pos.x < 0.0 || pos.x > bounds.x) {
                vel.x *= -0.8;
                pos.x = clamp(pos.x, 0.0, bounds.x);
            }
            if (pos.y < 0.0 || pos.y > bounds.y) {
                vel.y *= -0.8;
                pos.y = clamp(pos.y, 0.0, bounds.y);
            }
            
            // Apply damping
            vel *= damping;
            
            // Limit speed
            float speed = length(vel);
            if (speed > maxSpeed) {
                vel = normalize(vel) * maxSpeed;
            }
            
            outPosition = pos;
            outVelocity = vel;
        }
    )";
    
    // Particle render shader
    string particleVert = R"(
        #version 410
        
        layout(location = 0) in vec2 position;
        layout(location = 1) in vec2 velocity;
        
        uniform mat4 modelViewProjectionMatrix;
        uniform float particleSize;
        
        out float speed;
        
        void main() {
            gl_Position = modelViewProjectionMatrix * vec4(position, 0.0, 1.0);
            gl_PointSize = particleSize;
            speed = length(velocity) / 500.0;
        }
    )";
    
    string particleFrag = R"(
        #version 410
        
        in float speed;
        out vec4 fragColor;
        
        void main() {
            // Circular point
            vec2 coord = gl_PointCoord - vec2(0.5);
            if (length(coord) > 0.5) discard;
            
            // Color based on speed (blue to cyan)
            vec3 color = mix(vec3(0.2, 0.4, 0.8), vec3(0.4, 0.8, 1.0), clamp(speed, 0.0, 1.0));
            fragColor = vec4(color, 0.9);
        }
    )";
    
    // Line render shader with GEOMETRY SHADER
    string lineVert = R"(
        #version 410
        
        layout(location = 0) in vec2 position;
        
        out VS_OUT {
            vec2 position;
            int particleId;
        } vs_out;
        
        void main() {
            vs_out.position = position;
            vs_out.particleId = gl_VertexID;
        }
    )";
    
    string lineGeom = R"(
        #version 410
        
        layout(points) in;
        layout(line_strip, max_vertices = 256) out;
        
        in VS_OUT {
            vec2 position;
            int particleId;
        } gs_in[];
        
        uniform mat4 modelViewProjectionMatrix;
        uniform samplerBuffer particlePositions;  // All particle positions (in sorted order when sorting enabled)
        uniform isamplerBuffer sortedIndices;     // Original particle indices (for avoiding duplicates)
        uniform int numParticles;
        uniform float connectionDistance;
        uniform bool useSpatialSort;
        uniform int adaptiveSearchRadius;
        uniform bool showSortedConnections;
        
        out float lineAlpha;
        out vec3 lineColor;
        
        void main() {
            vec2 currentPos = gs_in[0].position;
            int mySortedPos = gs_in[0].particleId;  // This is now the sorted position
            int myOriginalId = texelFetch(sortedIndices, mySortedPos).r;  // Get original particle ID
            
            float maxDist = connectionDistance;
            float maxDistSq = maxDist * maxDist;
            
            int lineCount = 0;
            
            if (useSpatialSort) {
                // Optimized: Check nearby particles in sorted order
                int searchRadius = adaptiveSearchRadius;
                int startIdx = max(0, mySortedPos - searchRadius);
                int endIdx = min(numParticles - 1, mySortedPos + searchRadius);
                
                for (int sortedIdx = startIdx; sortedIdx <= endIdx && lineCount < 127; sortedIdx++) {
                    if (sortedIdx == mySortedPos) continue;  // Skip self
                    
                    // Get original ID to avoid duplicate lines
                    int otherOriginalId = texelFetch(sortedIndices, sortedIdx).r;
                    if (otherOriginalId <= myOriginalId) continue;  // Only draw i->j where j > i
                    
                    vec2 otherPos = texelFetch(particlePositions, sortedIdx).xy;
                    vec2 delta = otherPos - currentPos;
                    float distSq = dot(delta, delta);
                    
                    if (distSq < maxDistSq && distSq > 0.01) {
                        float dist = sqrt(distSq);
                        lineAlpha = 1.0 - (dist / maxDist);
                        
                        // Color lines based on distance in sorted array
                        if (showSortedConnections) {
                            float sortedDist = abs(float(sortedIdx - mySortedPos)) / float(searchRadius);
                            lineColor = mix(vec3(0.2, 0.8, 0.2), vec3(0.8, 0.2, 0.2), sortedDist);
                        } else {
                            lineColor = vec3(0.3, 0.6, 0.9);
                        }
                        
                        gl_Position = modelViewProjectionMatrix * vec4(currentPos, 0.0, 1.0);
                        EmitVertex();
                        
                        gl_Position = modelViewProjectionMatrix * vec4(otherPos, 0.0, 1.0);
                        EmitVertex();
                        
                        EndPrimitive();
                        lineCount++;
                    }
                }
            } else {
                // Brute force: Check all particles
                for (int i = 0; i < numParticles && lineCount < 127; i++) {
                    if (i == mySortedPos) continue;
                    
                    int otherOriginalId = texelFetch(sortedIndices, i).r;
                    if (otherOriginalId <= myOriginalId) continue;
                    
                    vec2 otherPos = texelFetch(particlePositions, i).xy;
                    vec2 delta = otherPos - currentPos;
                    float distSq = dot(delta, delta);
                    
                    if (distSq < maxDistSq && distSq > 0.01) {
                        float dist = sqrt(distSq);
                        lineAlpha = 1.0 - (dist / maxDist);
                        lineColor = vec3(0.3, 0.6, 0.9);
                        
                        gl_Position = modelViewProjectionMatrix * vec4(currentPos, 0.0, 1.0);
                        EmitVertex();
                        
                        gl_Position = modelViewProjectionMatrix * vec4(otherPos, 0.0, 1.0);
                        EmitVertex();
                        
                        EndPrimitive();
                        lineCount++;
                    }
                }
            }
        }
    )";
    
    string lineFrag = R"(
        #version 410
        
        in float lineAlpha;
        in vec3 lineColor;
        out vec4 fragColor;
        
        void main() {
            fragColor = vec4(lineColor, lineAlpha * 0.4);
        }
    )";
    
    // Compile shaders
    particleUpdateShader.setupShaderFromSource(GL_VERTEX_SHADER, updateVert);
    
    // Setup transform feedback
    const char* varyings[] = {"outPosition", "outVelocity"};
    GLuint program = particleUpdateShader.getProgram();
    glTransformFeedbackVaryings(program, 2, varyings, GL_INTERLEAVED_ATTRIBS);
    if (!particleUpdateShader.linkProgram()) {
        cout << "ERROR: Failed to link particle update shader!" << endl;
    } else {
        cout << "Particle update shader compiled successfully" << endl;
    }
    
    particleRenderShader.setupShaderFromSource(GL_VERTEX_SHADER, particleVert);
    particleRenderShader.setupShaderFromSource(GL_FRAGMENT_SHADER, particleFrag);
    if (!particleRenderShader.linkProgram()) {
        cout << "ERROR: Failed to link particle render shader!" << endl;
    } else {
        cout << "Particle render shader compiled successfully" << endl;
    }
    
    lineRenderShader.setupShaderFromSource(GL_VERTEX_SHADER, lineVert);
    lineRenderShader.setupShaderFromSource(GL_GEOMETRY_SHADER, lineGeom);
    lineRenderShader.setupShaderFromSource(GL_FRAGMENT_SHADER, lineFrag);
    if (!lineRenderShader.linkProgram()) {
        cout << "ERROR: Failed to link line render shader!" << endl;
    } else {
        cout << "Line render shader compiled successfully" << endl;
    }
}

void ofApp::setupParticles() {
    // Initialize particle data
    particles.resize(NUM_PARTICLES);
    particlesWithMorton.resize(NUM_PARTICLES);
    sortedIndices.resize(NUM_PARTICLES);
    
    cout << "Initializing " << NUM_PARTICLES << " particles..." << endl;
    
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particles[i].position.x = ofRandom(ofGetWidth());
        particles[i].position.y = ofRandom(ofGetHeight());
        particles[i].velocity.x = ofRandom(-50, 50);
        particles[i].velocity.y = ofRandom(-50, 50);
        sortedIndices[i] = i;  // Initially unsorted
    }
    
    cout << "First particle at: (" << particles[0].position.x << ", " << particles[0].position.y << ")" << endl;
    cout << "Last particle at: (" << particles[NUM_PARTICLES-1].position.x << ", " << particles[NUM_PARTICLES-1].position.y << ")" << endl;
    
    // Create ping-pong buffers for transform feedback
    glGenVertexArrays(2, vao);
    glGenBuffers(2, vbo);
    
    for (int i = 0; i < 2; i++) {
        glBindVertexArray(vao[i]);
        glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);
        
        // Upload initial data
        glBufferData(GL_ARRAY_BUFFER,
                    NUM_PARTICLES * sizeof(Particle),
                    particles.data(),
                    GL_DYNAMIC_COPY);
        
        // Position attribute
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                            sizeof(Particle), (void*)0);
        
        // Velocity attribute
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                            sizeof(Particle),
                            (void*)offsetof(Particle, velocity));
    }
    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    // Debug: Check VAO state
    cout << "VAO setup complete. Checking state..." << endl;
    glBindVertexArray(vao[0]);
    GLint enabled0, enabled1;
    glGetVertexAttribiv(0, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled0);
    glGetVertexAttribiv(1, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &enabled1);
    cout << "VAO[0]: Attribute 0 enabled: " << enabled0 << ", Attribute 1 enabled: " << enabled1 << endl;
    glBindVertexArray(0);
    
    // Create Texture Buffer Object for particle positions (used by geometry shader)
    glGenBuffers(1, &particlePositionTBO);
    glBindBuffer(GL_TEXTURE_BUFFER, particlePositionTBO);
    glBufferData(GL_TEXTURE_BUFFER, NUM_PARTICLES * sizeof(ofVec2f), nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    
    // Create texture handle for the TBO
    glGenTextures(1, &particlePositionTexture);
    glBindTexture(GL_TEXTURE_BUFFER, particlePositionTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, particlePositionTBO);
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    
    // Create Texture Buffer Object for sorted indices
    glGenBuffers(1, &sortedIndicesTBO);
    glBindBuffer(GL_TEXTURE_BUFFER, sortedIndicesTBO);
    glBufferData(GL_TEXTURE_BUFFER, NUM_PARTICLES * sizeof(int), sortedIndices.data(), GL_DYNAMIC_DRAW);
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
    
    // Create texture handle for sorted indices TBO
    glGenTextures(1, &sortedIndicesTexture);
    glBindTexture(GL_TEXTURE_BUFFER, sortedIndicesTexture);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, sortedIndicesTBO);  // Single integer per element
    glBindTexture(GL_TEXTURE_BUFFER, 0);
    
    // Setup proxy VBO for line rendering - initialize with actual position data
    particlePositionsOnly.resize(NUM_PARTICLES);
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particlePositionsOnly[i] = particles[i].position;
    }
    
    // Setup VBO with vertex data (ofVec2f positions)
    particleProxyVbo.setVertexData(particlePositionsOnly.data(), NUM_PARTICLES, GL_DYNAMIC_DRAW);
}

// Morton code (Z-order curve) functions for spatial sorting
uint32_t ofApp::expandBits(uint32_t v) {
    // Spread bits so there's a 0 between each bit
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

uint32_t ofApp::calculateMortonCode(float x, float y) {
    // Normalize to [0, 1024) range
    uint32_t xx = (uint32_t)(x / ofGetWidth() * 1024.0f);
    uint32_t yy = (uint32_t)(y / ofGetHeight() * 1024.0f);
    
    // Clamp to valid range
    xx = std::min(xx, 1023u);
    yy = std::min(yy, 1023u);
    
    // Interleave bits (Z-order curve)
    return expandBits(xx) | (expandBits(yy) << 1);
}

void ofApp::spatialSort() {
    // Calculate Morton codes for all particles
    for (int i = 0; i < NUM_PARTICLES; i++) {
        particlesWithMorton[i].index = i;
        particlesWithMorton[i].position = particles[i].position;
        particlesWithMorton[i].mortonCode = calculateMortonCode(
            particles[i].position.x,
            particles[i].position.y
        );
    }
    
    // Sort by Morton code (preserves spatial locality)
    std::sort(particlesWithMorton.begin(), particlesWithMorton.end(),
        [](const ParticleWithMorton& a, const ParticleWithMorton& b) {
            return a.mortonCode < b.mortonCode;
        });
    
    // Reorder positions in sorted order for rendering
    for (int i = 0; i < NUM_PARTICLES; i++) {
        sortedIndices[i] = particlesWithMorton[i].index;
        particlePositionsOnly[i] = particlesWithMorton[i].position;
    }
    
    // Upload sorted indices to GPU
    glBindBuffer(GL_TEXTURE_BUFFER, sortedIndicesTBO);
    glBufferSubData(GL_TEXTURE_BUFFER, 0, NUM_PARTICLES * sizeof(int), sortedIndices.data());
    glBindBuffer(GL_TEXTURE_BUFFER, 0);
}

void ofApp::updatePhysics() {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Update particles using transform feedback
    int readBuffer = currentBuffer;
    int writeBuffer = 1 - currentBuffer;
    
    particleUpdateShader.begin();
    particleUpdateShader.setUniform1f("deltaTime", timeStep);
    particleUpdateShader.setUniform1f("damping", damping);
    particleUpdateShader.setUniform2f("mousePos", mouseX, mouseY);
    particleUpdateShader.setUniform1f("mouseStrength",
                                     attractMode ? attractionStrength : -repulsionStrength);
    particleUpdateShader.setUniform2f("bounds", ofGetWidth(), ofGetHeight());
    particleUpdateShader.setUniform1f("maxSpeed", maxSpeed);
    
    // Bind read buffer
    glBindVertexArray(vao[readBuffer]);
    
    // Bind write buffer for transform feedback
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo[writeBuffer]);
    
    // Disable rasterization (we're only writing to buffer)
    glEnable(GL_RASTERIZER_DISCARD);
    
    // Perform transform feedback
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
    glEndTransformFeedback();
    
    glDisable(GL_RASTERIZER_DISCARD);
    
    particleUpdateShader.end();
    
    glBindVertexArray(0);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, 0);
    
    // Swap buffers
    currentBuffer = writeBuffer;
    
    // Read back particle positions for texture buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo[currentBuffer]);
    void* ptr = glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
    if (ptr) {
        memcpy(particles.data(), ptr, NUM_PARTICLES * sizeof(Particle));
        glUnmapBuffer(GL_ARRAY_BUFFER);
        
        // Extract positions and upload to texture buffer
        for (int i = 0; i < NUM_PARTICLES; i++) {
            particlePositionsOnly[i] = particles[i].position;
        }
        
        // Perform spatial sort if enabled (this will reorder particlePositionsOnly)
        if (useSpatialSort) {
            auto sortStart = std::chrono::high_resolution_clock::now();
            spatialSort();
            auto sortEnd = std::chrono::high_resolution_clock::now();
            float sortTime = std::chrono::duration<float, std::milli>(sortEnd - sortStart).count();
            avgSortTime = avgSortTime * 0.95f + sortTime * 0.05f;  // Exponential moving average
        } else {
            // If not sorting, just use original order for sorted indices
            for (int i = 0; i < NUM_PARTICLES; i++) {
                sortedIndices[i] = i;
            }
            // Upload unsorted indices
            glBindBuffer(GL_TEXTURE_BUFFER, sortedIndicesTBO);
            glBufferSubData(GL_TEXTURE_BUFFER, 0, NUM_PARTICLES * sizeof(int), sortedIndices.data());
            glBindBuffer(GL_TEXTURE_BUFFER, 0);
        }
        
        // Update texture buffer with positions (sorted or unsorted)
        glBindBuffer(GL_TEXTURE_BUFFER, particlePositionTBO);
        glBufferSubData(GL_TEXTURE_BUFFER, 0,
                       NUM_PARTICLES * sizeof(ofVec2f),
                       particlePositionsOnly.data());
        glBindBuffer(GL_TEXTURE_BUFFER, 0);
        
        // Update proxy VBO for geometry shader input
        particleProxyVbo.updateVertexData(particlePositionsOnly.data(), NUM_PARTICLES);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float physicsTime = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    avgPhysicsTime = avgPhysicsTime * 0.95f + physicsTime * 0.05f;
}

void ofApp::update() {
    updatePhysics();
    
    // Debug logging every 60 frames (once per second)
    if (frameCounter % 60 == 0) {
        cout << "Frame " << frameCounter << ": First particle at ("
             << particles[0].position.x << ", " << particles[0].position.y << ")" << endl;
    }
}

void ofApp::draw() {
    auto startRender = std::chrono::high_resolution_clock::now();
    
    ofEnableBlendMode(OF_BLENDMODE_ADD);
    
    // Draw lines using geometry shader (optional)
    if (drawLines) {
        lineRenderShader.begin();
        lineRenderShader.setUniformMatrix4f("modelViewProjectionMatrix",
                                           ofGetCurrentMatrix(OF_MATRIX_MODELVIEW) *
                                           ofGetCurrentMatrix(OF_MATRIX_PROJECTION));
        lineRenderShader.setUniform1i("numParticles", NUM_PARTICLES);
        lineRenderShader.setUniform1f("connectionDistance", connectionDistance);
        lineRenderShader.setUniform1i("useSpatialSort", useSpatialSort ? 1 : 0);
        lineRenderShader.setUniform1i("adaptiveSearchRadius", adaptiveSearchRadius);
        lineRenderShader.setUniform1i("showSortedConnections", showSortedConnections ? 1 : 0);
        
        // Bind texture buffer containing all particle positions
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, particlePositionTexture);
        lineRenderShader.setUniform1i("particlePositions", 0);
        
        // Bind texture buffer containing sorted indices
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_BUFFER, sortedIndicesTexture);
        lineRenderShader.setUniform1i("sortedIndices", 1);
        
        // Draw points - geometry shader will generate lines
        particleProxyVbo.draw(GL_POINTS, 0, NUM_PARTICLES);
        
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_BUFFER, 0);
        lineRenderShader.end();
    }
    
    // Draw particles
    if (useShaderRendering) {
        // GPU rendering with shaders
        glEnable(GL_PROGRAM_POINT_SIZE);  // Required for gl_PointSize to work
        
        // Debug: Check if we have valid data in the VBO
        if (frameCounter % 60 == 0) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo[currentBuffer]);
            Particle* testData = (Particle*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
            if (testData) {
                cout << "VBO data check - Particle 0: pos(" << testData[0].position.x << ", "
                     << testData[0].position.y << ") vel(" << testData[0].velocity.x << ", "
                     << testData[0].velocity.y << ")" << endl;
                cout << "Particle 100: pos(" << testData[100].position.x << ", "
                     << testData[100].position.y << ")" << endl;
                glUnmapBuffer(GL_ARRAY_BUFFER);
            }
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
        
        particleRenderShader.begin();
        
        // Use openFrameworks' built-in matrix
        ofMatrix4x4 mvp = ofGetCurrentMatrix(OF_MATRIX_PROJECTION) * ofGetCurrentMatrix(OF_MATRIX_MODELVIEW);
        particleRenderShader.setUniformMatrix4f("modelViewProjectionMatrix", mvp);
        particleRenderShader.setUniform1f("particleSize", particleSize);
        
        if (frameCounter % 60 == 0) {
            cout << "Drawing " << NUM_PARTICLES << " points with point size " << particleSize << endl;
        }
        
        glBindVertexArray(vao[currentBuffer]);
        
        // Check for OpenGL errors
        GLenum err = glGetError();
        if (err != GL_NO_ERROR && frameCounter % 60 == 0) {
            cout << "OpenGL error before draw: " << err << endl;
        }
        
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);
        
        err = glGetError();
        if (err != GL_NO_ERROR && frameCounter % 60 == 0) {
            cout << "OpenGL error after draw: " << err << endl;
        }
        
        glBindVertexArray(0);
        
        particleRenderShader.end();
        
        glDisable(GL_PROGRAM_POINT_SIZE);
    } else {
        // CPU rendering fallback (for debugging)
        ofSetColor(100, 150, 255);
        for (int i = 0; i < NUM_PARTICLES; i++) {
            ofDrawCircle(particles[i].position.x, particles[i].position.y, particleSize);
        }
    }
    
    ofDisableBlendMode();
    
    // Additional debug: Try drawing a single point with immediate mode
    if (useShaderRendering && frameCounter < 120) {
        glPointSize(20.0f);
        glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
        glBegin(GL_POINTS);
        glVertex2f(ofGetWidth() / 2, ofGetHeight() / 2);
        glEnd();
        
        ofSetColor(255, 255, 0);
        ofDrawBitmapString("GPU Mode - If you see a RED point, fixed-function works", 20, ofGetHeight() - 40);
    }
    
    // Debug: Draw a test circle to verify rendering is working
    if (frameCounter < 60) {  // Only for first second
        ofSetColor(255, 0, 0);
        ofDrawCircle(ofGetWidth() / 2, ofGetHeight() / 2, 50);
    }
    
    auto endRender = std::chrono::high_resolution_clock::now();
    float renderTime = std::chrono::duration<float, std::milli>(endRender - startRender).count();
    avgRenderTime = avgRenderTime * 0.95f + renderTime * 0.05f;
    
    // Draw info overlay
    ofSetColor(255);
    stringstream ss;
    ss << "FPS: " << (int)ofGetFrameRate() << " (" << ofGetLastFrameTime() * 1000.0f << "ms)" << endl;
    ss << "Particles: " << NUM_PARTICLES << endl;
    ss << endl;
    ss << "PERFORMANCE:" << endl;
    ss << "  Physics + Sort: " << avgPhysicsTime << "ms" << endl;
    if (useSpatialSort) {
        ss << "    - Sort time: " << avgSortTime << "ms" << endl;
    }
    ss << "  Rendering: " << avgRenderTime << "ms" << endl;
    ss << "  Total: " << (avgPhysicsTime + avgRenderTime) << "ms" << endl;
    ss << endl;
    ss << "MODE:" << endl;
    ss << "  Physics: " << (attractMode ? "ATTRACT" : "REPEL") << endl;
    ss << "  Rendering: " << (useShaderRendering ? "GPU" : "CPU") << endl;
    ss << "  Draw Lines: " << (drawLines ? "ON" : "OFF") << endl;
    ss << "  Spatial Sort: " << (useSpatialSort ? "ON" : "OFF") << endl;
    if (useSpatialSort) {
        ss << "  Search Radius: " << adaptiveSearchRadius << endl;
    }
    ss << "  Morton Colors: " << (showMortonColors ? "ON" : "OFF") << endl;
    ss << "  Sorted Connections: " << (showSortedConnections ? "ON" : "OFF") << endl;
    ss << endl;
    ss << "CONTROLS:" << endl;
    ss << "  SPACE: Toggle attract/repel" << endl;
    ss << "  R: Toggle GPU/CPU rendering" << endl;
    ss << "  L: Toggle line rendering" << endl;
    ss << "  S: Toggle spatial sort" << endl;
    ss << "  M: Toggle Morton color visualization" << endl;
    ss << "  C: Toggle sorted connection colors" << endl;
    ss << "  +/-: Adjust search radius (" << adaptiveSearchRadius << ")" << endl;
    ss << "  [/]: Adjust connection distance (" << (int)connectionDistance << ")";
    ofDrawBitmapString(ss.str(), 20, 20);
    
    frameCounter++;
}

void ofApp::keyPressed(int key) {
    if (key == ' ') {
        attractMode = !attractMode;
    }
    if (key == 'r' || key == 'R') {
        useShaderRendering = !useShaderRendering;
        cout << "Rendering mode: " << (useShaderRendering ? "GPU" : "CPU") << endl;
    }
    if (key == 'l' || key == 'L') {
        drawLines = !drawLines;
        cout << "Line rendering: " << (drawLines ? "ON" : "OFF") << endl;
    }
    if (key == 's' || key == 'S') {
        useSpatialSort = !useSpatialSort;
        cout << "Spatial sort: " << (useSpatialSort ? "ON" : "OFF") << endl;
    }
    if (key == 'm' || key == 'M') {
        showMortonColors = !showMortonColors;
        cout << "Morton color visualization: " << (showMortonColors ? "ON" : "OFF") << endl;
    }
    if (key == 'c' || key == 'C') {
        showSortedConnections = !showSortedConnections;
        cout << "Sorted connection colors: " << (showSortedConnections ? "ON" : "OFF") << endl;
    }
    if (key == '+' || key == '=') {
        adaptiveSearchRadius = std::min(adaptiveSearchRadius + 50, NUM_PARTICLES);
        cout << "Search radius: " << adaptiveSearchRadius << endl;
    }
    if (key == '-' || key == '_') {
        adaptiveSearchRadius = std::max(adaptiveSearchRadius - 50, 50);
        cout << "Search radius: " << adaptiveSearchRadius << endl;
    }
    if (key == '[') {
        connectionDistance = fmax(connectionDistance - 10.0f, 20.0f);
        cout << "Connection distance: " << connectionDistance << endl;
    }
    if (key == ']') {
        connectionDistance = fmin(connectionDistance + 10.0f, 200.0f);
        cout << "Connection distance: " << connectionDistance << endl;
    }
}
