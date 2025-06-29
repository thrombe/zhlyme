#version 460

#include <common.glsl>
#include <uniforms.glsl>

struct GpuState {
    int particle_count;
    int seed_id;

    int bad_flag;

    vec4 _pad_aligned;
};

vec3 quad_verts[6] = vec3[6](
    vec3(1.0, 1.0, 0.0),
    vec3(-1.0, 1.0, 0.0),
    vec3(1.0, -1.0, 0.0),
    vec3(1.0, -1.0, 0.0),
    vec3(-1.0, 1.0, 0.0),
    vec3(-1.0, -1.0, 0.0)
);
vec2 quad_uvs[6] = vec2[6](
    vec2(1.0, 1.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 0.0),
    vec2(0.0, 1.0),
    vec2(0.0, 0.0)
);

#ifdef COMPUTE_PASS
    #define bufffer buffer
#else
    #define bufffer readonly buffer
#endif

layout(set = 0, binding = _bind_camera) uniform Ubo {
    Uniforms ubo;
};
layout(set = 0, binding = _bind_scratch) bufffer ScratchBuffer {
    GpuState state;
};
layout(set = 0, binding = _bind_particle_types) bufffer ParticleTypeBuffer {
    ParticleType particle_types[];
};
layout(set = 0, binding = _bind_particle_force_matrix) bufffer ParticleForceMatrixBuffer {
    ParticleForce particle_force_matrix[];
};
layout(set = 0, binding = _bind_particles_back) bufffer ParticleBackBuffer {
    Particle particles_back[];
};
layout(set = 0, binding = _bind_particles) bufffer ParticleBuffer {
    Particle particles[];
};
layout(set = 0, binding = _bind_particle_bins_back) bufffer ParticleBinBackBuffer {
    int particle_bins_back[];
};
layout(set = 0, binding = _bind_particle_bins) bufffer ParticleBinBuffer {
    int particle_bins[];
};
layout(set = 0, binding = _bind_particles_draw_call) bufffer ParticlesDrawCallBuffer {
    DrawCall draw_call;
};
layout(push_constant) uniform PushConstantsUniform {
    PushConstants push;
};

void set_seed(int id) {
    seed = int(ubo.frame.frame) ^ id ^ floatBitsToInt(ubo.frame.time) ^ push.seed;
}

#ifdef SPAWN_PARTICLES_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;
        set_seed(id);

        if (id == 0) {
            int count = int(state.particle_count);
            draw_call.index_count = count * 6;
            draw_call.instance_count = 1;
            draw_call.first_index = 0;
            draw_call.vertex_offset = 0;
            draw_call.first_instance = 0;
        }

        if (id >= ubo.params.spawn_count) {
            return;
        }

        vec2 mres = vec2(ubo.frame.monitor_width, ubo.frame.monitor_height);
        int index = atomicAdd(state.particle_count, 1);
        Particle p;
        p.pos = vec3(random(), random(), random()) * vec3(float(ubo.params.world_size_x), float(ubo.params.world_size_y), float(ubo.params.world_size_z));
        p.vel = 50.0 * (vec3(random(), random(), random()) - 0.5) * 2.0;
        p.type_index = clamp(int(random() * ubo.params.particle_type_count), 0, ubo.params.particle_type_count - 1);
        particles[index] = p;
    }
#endif // SPAWN_PARTICLES_PASS

#ifdef BIN_RESET_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;

        // 1 larger then the buffer to store capacities
        if (id > ubo.params.bin_buf_size) {
            return;
        }

        particle_bins[id] = 0;
        particle_bins_back[id] = 0;

        if (id == 0) {
            state.bad_flag = 0;
        }
    }
#endif // BIN_RESET_PASS

#ifdef PARTICLE_COUNT_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;

        if (id >= state.particle_count) {
            return;
        }

        Particle p = particles[id];

        ivec3 pos = ivec3(p.pos / ubo.params.bin_size);
        int index = clamp(pos.z * ubo.params.bin_buf_size_y * ubo.params.bin_buf_size_x + pos.y * ubo.params.bin_buf_size_x + pos.x, 0, ubo.params.bin_buf_size);

        int _count = atomicAdd(particle_bins_back[index], 1);
    }
#endif // PARTICLE_COUNT_PASS

#ifdef BIN_PREFIX_SUM_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;

        // 1 larger then the buffer to store capacities
        if (id > ubo.params.bin_buf_size) {
            return;
        }

        int step = 1 << push.reduce_step;
        if (id >= step) {
            int a = particle_bins_back[id];
            int b = particle_bins_back[id - step];
            particle_bins[id] = a + b;
        } else {
            int a = particle_bins_back[id];
            particle_bins[id] = a;
        }
    }
#endif // BIN_PREFIX_SUM_PASS

#ifdef PARTICLE_BINNING_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;
        set_seed(id);

        if (id >= state.particle_count) {
            return;
        }
        vec3 world = vec3(float(ubo.params.world_size_x), float(ubo.params.world_size_y), float(ubo.params.world_size_z));

        Particle p = particles[id];

        f32 vel = length(p.vel);
        f32 dist = 2.0 * length(p.pos - world / 2.0) / length(world);
        f32 particle_entropy = 0.0;
        particle_entropy += float(vel < 10.0) * 0.001 + float(vel > 20.0) * 0.001;
        particle_entropy += sqrt(p.exposure) * 0.0001;
        particle_entropy += float(p.age > 1000.0) * 0.0003;
        particle_entropy *= ubo.params.entropy;

        bool killed = false;
        if (particle_entropy > random()) {
            killed = true;
        }

        if (ubo.params.randomize_particle_types != 0 || killed) {
            p.type_index = randuint() % ubo.params.particle_type_count;
            p.age = 0.0;
            p.exposure = 0.0;
        }
        if (ubo.params.randomize_particle_attrs != 0 || killed) {
            p.pos = vec3(random(), random(), random()) * world;
            p.vel = (vec3(random(), random(), random()) - 0.5) * mix(2000, 2, killed);
        }

        ivec3 pos = ivec3(p.pos / ubo.params.bin_size);
        int index = clamp(pos.z * ubo.params.bin_buf_size_y * ubo.params.bin_buf_size_x + pos.y * ubo.params.bin_buf_size_x + pos.x, 0, ubo.params.bin_buf_size);

        int bin_index = atomicAdd(particle_bins[index], -1);

        particles_back[bin_index - 1] = p;
    }
#endif // PARTICLE_BINNING_PASS

#ifdef TICK_PARTICLES_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;
        set_seed(id);

        if (id >= state.particle_count) {
            return;
        }

        // 2 ways to compute forces
        //  - first is to compute and store forces on particles in 1 pass and update vel, pos in another
        //    - it's possible to have different radius for forces, and check only the bins that would actually matter.
        //    - less friendly to gpus cuz different cores in a warp need different amounts of compute.
        //  - second is to assume a max influence distance and check all in the range.
        //    - if we have a max radius to forces, we end up wasting some checks when (max radius across all types >> max radius for 1 type)

        Particle p = particles_back[id];
        // ParticleType pt = particle_types[p.type_index];

        ivec3 bpos = ivec3(p.pos / ubo.params.bin_size);
        ivec3 bworld = ivec3(ubo.params.bin_buf_size_x, ubo.params.bin_buf_size_y, ubo.params.bin_buf_size_z);

        ivec3 world = ivec3(ubo.params.world_size_x, ubo.params.world_size_y, ubo.params.world_size_z);

        vec3 fattract = vec3(0.0);
        vec3 fcollide = vec3(0.0);
        f32 exposure = 0.0;
        for (int z = -1; z <= 1; z++) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    ivec3 bpos = (ivec3(x, y, z) + bpos + bworld) % bworld;
                    int index = bpos.z * bworld.y * bworld.x + bpos.y * bworld.x + bpos.x;
                    int offset_start = particle_bins[index];
                    int offset_end = particle_bins[index + 1];

                    for (int i = offset_start; i < offset_end; i++) {
                        if (i == id) {
                            continue;
                        }

                        Particle o = particles_back[i];
                        // ParticleType ot = particle_types[o.type_index];

                        ParticleForce forces = particle_force_matrix[p.type_index * ubo.params.particle_type_count + o.type_index];

                        // Calculate wrapped distance
                        vec3 dir = o.pos - p.pos;
                        dir -= world * sign(dir) * vec3(greaterThanEqual(abs(dir), world * 0.5));

                        f32 dist = length(dir);
                        if (dist <= 0.0) {
                            continue;
                        }

                        exposure += 1.0;

                        dir /= dist;

                        f32 bin_size = ubo.params.bin_size;
                        f32 collision_r = forces.collision_radius * bin_size;
                        f32 collision_s = forces.collision_strength * ubo.params.collision_strength_scale;
                        f32 attraction_r = forces.attraction_radius * bin_size;
                        f32 attraction_peak_r = mix(forces.collision_radius, forces.attraction_radius, forces.attraction_peak_dist_factor) * bin_size;
                        f32 attraction_s = forces.attraction_strength * ubo.params.attraction_strength_scale;
                        if (dist < collision_r) {
                            fcollide -= collision_s * (1.0 - dist / collision_r) * dir;
                        } else if (dist < attraction_peak_r) {
                            fattract += attraction_s * ((dist - collision_r) / (attraction_peak_r - collision_r)) * dir;
                        } else if (dist < attraction_r) {
                            fattract += attraction_s * (1.0 - (dist - attraction_peak_r) / (attraction_r - attraction_peak_r)) * dir;
                        } else {
                            exposure -= 1.0;
                        }
                    }
                }
            }
        }

        f32 flen = length(fattract);
        // pforce *= flen / (flen + 1);
        // fattract *= 1.0/log(flen + 1);
        // fattract *= pow(flen, 0.83) / max(flen, 1);
        fattract *= min(flen, ubo.params.max_attraction_factor * ubo.params.attraction_strength_scale)/max(flen, 1);

        vec3 pforce = fcollide + fattract;
        p.vel *= ubo.params.friction;
        p.vel += pforce * ubo.params.delta;
        p.pos += p.vel * ubo.params.delta;

        // position wrapping
        p.pos += world * vec3(lessThan(p.pos, vec3(0)));
        p.pos -= world * vec3(greaterThanEqual(p.pos, world));

        // prevents position blow up
        p.pos = clamp(p.pos, vec3(0.0), world);

        p.age += 1.0;
        p.exposure = exposure;

        particles[id] = p;
    }
#endif // TICK_PARTICLES_PASS

#ifdef RENDER_VERT_PASS
    layout(location = 0) out vec4 vcolor;
    layout(location = 1) out vec2 vuv;
    layout(location = 2) out f32 z_factor;
    void main() {
        int particle_index = gl_VertexIndex / 6;
        int vert_index = gl_VertexIndex % 6;

        Particle p = particles[particle_index];
        ParticleType t = particle_types[p.type_index];
        vec2 vpos = quad_verts[vert_index].xy;

        float zoom = ubo.params.zoom;
        float particle_size = t.particle_scale * ubo.params.particle_visual_size;
        vec2 mres = vec2(ubo.frame.monitor_width, ubo.frame.monitor_height);
        vec2 wres = vec2(ubo.frame.width, ubo.frame.height);

        z_factor = abs(p.pos.z - ubo.params.world_size_z * 0.5) / max(ubo.params.world_size_z * 0.5, 1);
        f32 z_shrink = (1.0 - ubo.params.particle_z_shrinking_factor) + z_factor * ubo.params.particle_z_shrinking_factor;
        z_shrink = clamp(z_shrink, 0, 1);

        vec2 pos = p.pos.xy + ubo.camera.eye.xy - vec2(float(ubo.params.world_size_x), float(ubo.params.world_size_y)) * 0.5;
        pos += vpos * 0.5 * particle_size * z_shrink;
        pos /= mres; // world space to 0..1
        pos *= mres/wres; // 0..1 scaled wrt window size
        pos *= zoom;
        pos *= 2.0;
        gl_Position = vec4(pos, 0.0, 1.0);

        vcolor = t.color;
        vuv = quad_uvs[vert_index];
    }
#endif // RENDER_VERT_PASS

#ifdef RENDER_FRAG_PASS
    layout(location = 0) in vec4 vcolor;
    layout(location = 1) in vec2 vuv;
    layout(location = 2) in f32 z_factor;
    layout(location = 0) out vec4 fcolor;
    void main() {
        float zoom = ubo.params.zoom;
        float distanceFromCenter = length(vuv.xy - 0.5);
        float mask = 1.0 - smoothstep(0.5 - z_factor * ubo.params.particle_z_blur_factor - 0.1/zoom, 0.5, distanceFromCenter);
        // mask = pow(1.0 - distanceFromCenter, 4.5) * mask;
        fcolor = vec4(vcolor.xyz, vcolor.a * mask * (0.4 +  0.6 * (1.0 - z_factor)));
    }
#endif // RENDER_FRAG_PASS

#ifdef BG_VERT_PASS
    void main() {
        vec3 pos = quad_verts[gl_VertexIndex];

        pos.z = 1.0 - 0.000001;

        gl_Position = vec4(pos, 1.0);
    }
#endif // BG_VERT_PASS

#ifdef BG_FRAG_PASS
    layout(location = 0) out vec4 fcolor;
    void main() {
        float grid_size = ubo.params.grid_size;
        float zoom = ubo.params.zoom;
        vec2 eye = ubo.camera.eye.xy;
        vec2 mres = vec2(ubo.frame.monitor_width, ubo.frame.monitor_height);
        vec2 wres = vec2(ubo.frame.width, ubo.frame.height);

        vec2 coord = gl_FragCoord.xy;
        coord -= wres / 2.0;
        coord /= zoom;
        coord -= eye;
        coord /= grid_size;

        vec2 rounded = vec2(floor(coord.x), floor(coord.y));
        float checker = mod(floor(rounded.x) + floor(rounded.y), 2.0);

        vec3 color = mix(vec3(0.2, 0.15, 0.35), vec3(0.25, 0.20, 0.40), checker);

        // debug renderr `particle_bins`
        // ivec2 pos = ivec2(int(coord.x), int(coord.y) + 3);
        // int index = pos.y * ubo.frame.width + pos.x;
        // if (ubo.params.bin_buf_size > index && index >= 0) {
        //     color = vec3(float(particle_bins[index] > ubo.params.particle_count * mod(ubo.frame.time, 1)));
        // }

        // set bad_flag to 1 for debugging
        if (state.bad_flag > 0) {
            color = vec3(1, 0, 0);
        }
        
        fcolor = vec4(color, 1.0);
    }
#endif // BG_FRAG_PASS
