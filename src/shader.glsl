#version 460

#include <common.glsl>
#include <uniforms.glsl>

struct GpuState {
    int ant_count;
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

layout(set = _set_render, binding = _bind_camera) uniform Ubo {
    Uniforms ubo;
};
layout(set = _set_render, binding = _bind_scratch) bufffer ScratchBuffer {
    GpuState state;
};
layout(set = _set_render, binding = _bind_ant_types) bufffer AntTypeBuffer {
    AntType ant_types[];
};
layout(set = _set_render, binding = _bind_ants_back) bufffer AntBackBuffer {
    Ant ants_back[];
};
layout(set = _set_render, binding = _bind_ants) bufffer AntBuffer {
    Ant ants[];
};
layout(set = _set_ant_bins, binding = _bind_ant_bins_back) bufffer AntBinBackBuffer {
    int ant_bins_back[];
};
layout(set = _set_ant_bins, binding = _bind_ant_bins) bufffer AntBinBuffer {
    int ant_bins[];
};
layout(set = _set_pheromones, binding = _bind_pheromones_back) bufffer PheromoneBackBuffer {
    PheromoneBin pheromones_back[];
};
layout(set = _set_pheromones, binding = _bind_pheromones) bufffer PheromoneBuffer {
    PheromoneBin pheromones[];
};
layout(set = _set_render, binding = _bind_ants_draw_call) bufffer AntsDrawCallBuffer {
    DrawCall draw_call;
};

#if defined(BIN_PREFIX_SUM_PASS)
    layout(push_constant) uniform PushConstantsReduce_ {
        PushConstantsReduce push;
    };
#elif defined(SPREAD_PHEROMONES_PASS)
    layout(push_constant) uniform PushConstantsBlur_ {
        PushConstantsBlur push;
    };
#else
    layout(push_constant) uniform PushConstantsCompute_ {
        PushConstantsCompute push;
    };
#endif

void set_seed(int id) {
    seed = int(ubo.frame.frame) ^ id ^ floatBitsToInt(ubo.frame.time) ^ push.seed;
}

#ifdef SPAWN_ANTS_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;
        set_seed(id);

        if (id == 0) {
            int count = int(state.ant_count);
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
        int index = atomicAdd(state.ant_count, 1);
        Ant p;
        p.pos = vec2(random(), random()) * vec2(float(ubo.params.world_size_x), float(ubo.params.world_size_y));
        p.vel = 50.0 * (vec2(random(), random()) - 0.5) * 2.0;
        p.type_index = clamp(int(random() * ubo.params.ant_type_count), 0, ubo.params.ant_type_count - 1);
        ants[index] = p;
    }
#endif // SPAWN_ANTS_PASS

#ifdef BIN_RESET_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;

        // 1 larger then the buffer to store capacities
        if (id > ubo.params.bin_buf_size) {
            return;
        }

        ant_bins[id] = 0;
        ant_bins_back[id] = 0;

        if (id == 0) {
            state.bad_flag = 0;
        }
    }
#endif // BIN_RESET_PASS

#ifdef COUNT_ANTS_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;

        if (id >= state.ant_count) {
            return;
        }

        Ant p = ants[id];

        ivec2 pos = ivec2(p.pos / ubo.params.bin_size);
        int index = clamp(pos.y * ubo.params.bin_buf_size_x + pos.x, 0, ubo.params.bin_buf_size);

        int _count = atomicAdd(ant_bins_back[index], 1);
    }
#endif // COUNT_ANTS_PASS

#ifdef BIN_PREFIX_SUM_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;

        // 1 larger then the buffer to store capacities
        if (id > ubo.params.bin_buf_size) {
            return;
        }

        int step = 1 << push.step;
        if (id >= step) {
            int a = ant_bins_back[id];
            int b = ant_bins_back[id - step];
            ant_bins[id] = a + b;
        } else {
            int a = ant_bins_back[id];
            ant_bins[id] = a;
        }
    }
#endif // BIN_PREFIX_SUM_PASS

#ifdef BIN_ANTS_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;
        set_seed(id);

        if (id >= state.ant_count) {
            return;
        }

        Ant p = ants[id];

        ivec2 pos = ivec2(p.pos / ubo.params.bin_size);
        int index = clamp(pos.y * ubo.params.bin_buf_size_x + pos.x, 0, ubo.params.bin_buf_size);

        int bin_index = atomicAdd(ant_bins[index], -1);

        // NOTE: do this stuff *after* we have the bin index. we need to respect the prefix sum bin data to get useful index values.
        //  this is mostly implemented as a hack. after resetting/killing a ant - it likely won't interact with anything until the next frame
        //  but that is fine if we save an entire pass (+ ant buffer copy) especially for this, which would be the proper way to implement this.
        {
            // entropy calculation
            {
                vec2 world = vec2(float(ubo.params.world_size_x), float(ubo.params.world_size_y));
                f32 vel = length(p.vel);
                f32 dist = 2.0 * length(p.pos - world / 2.0) / length(world);
                f32 ant_entropy = 0.0;
                ant_entropy += float(vel < 10.0) * 0.001 + float(vel > 20.0) * 0.001;
                ant_entropy += sqrt(p.exposure) * 0.0001;
                ant_entropy += float(p.age > 1000.0) * 0.0003;
                ant_entropy *= ubo.params.entropy;

                // framerate and step independent entropy
                ant_entropy *= ubo.params.delta * 100.0;

                if (ant_entropy > random()) {
                    p.pos = vec2(random(), random()) * world;
                    p.vel = (vec2(random(), random()) - 0.5) * 2;
                    p.type_index = randuint() % ubo.params.ant_type_count;
                    p.age = 0.0;
                    p.exposure = 0.0;
                }
            }

            // randomize ants
            {
                if (ubo.params.randomize_ant_types != 0) {
                    p.type_index = randuint() % ubo.params.ant_type_count;
                    p.age = 0.0;
                    p.exposure = 0.0;
                }
                if (ubo.params.randomize_ant_attrs != 0) {
                    vec2 world = vec2(float(ubo.params.world_size_x), float(ubo.params.world_size_y));
                    p.pos = vec2(random(), random()) * world;
                    p.vel = (vec2(random(), random()) - 0.5) * 2000;
                }
            }
        }

        ants_back[bin_index - 1] = p;
    }
#endif // BIN_ANTS_PASS

#ifdef TICK_ANTS_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        int id = global_id;
        set_seed(id);

        if (id >= state.ant_count) {
            return;
        }

        ivec2 world = ivec2(ubo.params.world_size_x, ubo.params.world_size_y);
        ivec2 bworld = ivec2(ubo.params.bin_buf_size_x, ubo.params.bin_buf_size_y);

        Ant p = ants_back[id];
        AntType pt = ant_types[p.type_index];

        f32 rad = ubo.params.max_pheromone_detection_radius * pt.pheromone_detection_radius;
        vec2 pdir = vec2(0.0);
        for (i32 y = int(floor(-rad)); y <= int(ceil(rad)); y++) {
            for (i32 x = int(floor(-rad)); x <= int(ceil(rad)); x++) {
                vec2 dir = vec2(x, y);
                ivec2 pos = ivec2(p.pos);
                pos += ivec2(normalize(p.vel) * ubo.params.max_pheromone_detection_distance * pt.pheromone_detection_distance);
                pos += ivec2(dir);

                if (ubo.params.world_wrapping == 1) {
                    pos = ivec2(pos.x % world.x, pos.y % world.y);
                }

                if (length(dir) > rad) {
                    continue;
                }

                if (length(dir) > 0.0001) {
                    dir /= length(dir);
                }

                if (ubo.params.world_wrapping == 0) {
                    if (pos.x < 0 || pos.x >= world.x || pos.y < 0 || pos.y >= world.y) {
                        pdir -= dir;
                        continue;
                    }
                }

                if (dot(p.vel, dir) * pt.pheromone_attraction >= 0.0) {
                    pdir += dir * pheromones_back[pos.y * world.x + pos.x].strength * pt.pheromone_attraction * ubo.params.pheromone_attraction;
                }
            }
        }

        int index = int(p.pos.y) * world.x + int(p.pos.x);
        f32 pheromone = pheromones_back[index].strength;

        p.vel += pdir * 100.0 * ubo.params.delta;
        // TODO: i'll prob have to use some continuous noise functions to make this turning time step independent
        // random wander turning
        p.vel += (vec2(random(), random()) - 0.5) * pt.wander_strength * ubo.params.max_wander_strength * 100.0 * ubo.params.delta;
        p.vel = normalize(p.vel) * ubo.params.ant_velocity;

        ivec2 bpos = ivec2(p.pos / ubo.params.bin_size);

        vec2 fcollide = vec2(0.0);
        f32 exposure = 0.0;
        if (ubo.params.collision_strength_scale > 0.001 && ubo.params.collision_radius_scale > 0.001) {
            for (int y = -1; y <= 1; y++) {
                for (int x = -1; x <= 1; x++) {
                    ivec2 bpos = (ivec2(x, y) + bpos + bworld) % bworld;
                    int index = bpos.y * bworld.x + bpos.x;
                    int offset_start = ant_bins[index];
                    int offset_end = ant_bins[index + 1];

                    for (int i = offset_start; i < offset_end; i++) {
                        if (i == id) {
                            continue;
                        }

                        Ant o = ants_back[i];
                        AntType ot = ant_types[o.type_index];

                        // Calculate wrapped distance
                        vec2 dir = o.pos - p.pos;
                        dir -= world * sign(dir) * vec2(greaterThanEqual(abs(dir), world * 0.5));

                        f32 dist = length(dir);
                        if (dist <= 0.0) {
                            continue;
                        }

                        dir /= dist;

                        f32 bin_size = ubo.params.bin_size;
                        f32 collision_r = max(pt.collision_radius, ot.collision_radius) * ubo.params.collision_radius_scale * bin_size;
                        f32 collision_s = ubo.params.collision_strength_scale * max(pt.collision_strength, ot.collision_strength);
                        if (dist < collision_r) {
                            fcollide -= collision_s * (1.0 - dist / collision_r) * dir;
                            exposure += 1.0;
                        }
                    }
                }
            }
        }

        vec2 pforce = fcollide;
        p.vel += pforce * ubo.params.delta;
        p.pos += p.vel * ubo.params.delta;

        if (ubo.params.world_wrapping == 1) {
            // position wrapping
            p.pos += world * vec2(lessThan(p.pos, vec2(0)));
            p.pos -= world * vec2(greaterThanEqual(p.pos, world));
        } else {
            if (p.pos.x < 0 || p.pos.x >= world.x) {
                p.vel.x *= -1;
            }
            if (p.pos.y < 0 || p.pos.y >= world.y) {
                p.vel.y *= -1;
            }
        }

        // prevents position blow up
        p.pos = clamp(p.pos, vec2(0.0), world);

        // TODO: ants race to set this value. but whatever man
        pheromones[index].strength = min(ubo.params.max_pheromone_strength, pheromone + pt.pheromone_strength);
        pheromones[index].type = int(p.type_index);

        p.age += 100.0 * ubo.params.delta;
        p.exposure = exposure * 100.0 * ubo.params.delta;

        ants[id] = p;
    }
#endif // TICK_ANTS_PASS

#ifdef SPREAD_PHEROMONES_PASS
    layout (local_size_x = 8, local_size_y = 8) in;
    void main() {
        ivec2 id = ivec2(gl_GlobalInvocationID.xy);

        ivec2 world = ivec2(ubo.params.world_size_x, ubo.params.world_size_y);
        if (id.x >= world.x || id.y >= world.y) {
            return;
        }

        PheromoneBin npb;
        npb.strength = 0.0;
        npb.type = -1;
        f32 count = 0.0;
        int spread = ubo.params.spread_half_size;
        for (int t = -spread; t <= spread; t++) {
            ivec2 pos = ivec2(0);

            if (ubo.params.world_wrapping == 1) {
                if (push.dimension == 0) {
                    pos = ivec2(id.x, (world.y + id.y + t) % world.y);
                } else {
                    pos = ivec2((world.x + id.x + t) % world.x, id.y);
                }
            } else {
                if (push.dimension == 0) {
                    pos = ivec2(id.x, id.y + t);
                } else {
                    pos = ivec2(id.x + t, id.y);
                }

                if (pos.x < 0 || pos.x >= world.x || pos.y < 0 || pos.y >= world.y) {
                    continue;
                }
            }
            
            PheromoneBin pb = pheromones_back[pos.y * world.x + pos.x];
            // TODO: maybe also bind 'pheromones_back' as an image and do the fancy texture sampling blur
            npb.strength += pb.strength;
            count += 1.0;
            if ((npb.type < 0 || random() > 0.5) && pb.type >= 0) {
                npb.type = pb.type;
            }
        }
        npb.strength /= count;

        PheromoneBin opb = pheromones_back[id.y * world.x + id.x];
        npb.strength = mix(opb.strength, npb.strength, 10.0 * ubo.params.delta);

        if (push.dimension == 1) {
            npb.strength = max(0.0, npb.strength - ubo.params.pheromone_fade * ubo.params.delta);
        }
        if (npb.type < 0 || (random() > 0.5 && opb.type >= 0)) {
            npb.type = opb.type;
        }
        if (abs(npb.strength) < 0.0001) {
            npb.type = -1;
        }

        pheromones[id.y * world.x + id.x] = npb;
    }
#endif // SPREAD_PHEROMONES_PASS

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

        vec3 color = mix(vec3(80, 90, 80)/255.0, vec3(120, 124, 100)/255.0, checker);

        // debug renderr `ant_bins`
        // ivec2 pos = ivec2(int(coord.x), int(coord.y) + 3);
        // int index = pos.y * ubo.frame.width + pos.x;
        // if (ubo.params.bin_buf_size > index && index >= 0) {
        //     color = vec3(float(ant_bins[index] > ubo.params.ant_count * mod(ubo.frame.time, 1)));
        // }

        // set bad_flag to 1 for debugging
        if (state.bad_flag > 0) {
            color = vec3(1, 0, 0);
        }
        
        fcolor = vec4(color, 1.0);
    }
#endif // BG_FRAG_PASS

#ifdef RENDER_PHEROMONES_VERT_PASS
    void main() {
        vec3 pos = quad_verts[gl_VertexIndex];

        pos.z = 1.0 - 0.000001;

        gl_Position = vec4(pos, 1.0);
    }
#endif // RENDER_PHEROMONES_VERT_PASS

#ifdef RENDER_PHEROMONES_FRAG_PASS
    layout(location = 0) out vec4 fcolor;
    void main() {
        float grid_size = ubo.params.grid_size;
        float zoom = ubo.params.zoom;
        vec2 eye = ubo.camera.eye.xy;
        vec2 mres = vec2(ubo.frame.monitor_width, ubo.frame.monitor_height);
        vec2 wres = vec2(ubo.frame.width, ubo.frame.height);
        ivec2 world = ivec2(ubo.params.world_size_x, ubo.params.world_size_y);

        vec2 coord = gl_FragCoord.xy;
        coord -= wres / 2.0;
        coord /= zoom;
        coord += vec2(world) / 2.0; coord -= eye;

        int index = int(coord.y) * world.x + int(coord.x);
        if (coord.x > 0 && coord.y > 0 && coord.x < world.x && coord.y < world.y && index >=0 && index < world.x * world.y) {
            PheromoneBin pb = pheromones[index];
            vec4 color = vec4(1.0);
            if (pb.type >= 0) {
                color = ant_types[pb.type].color;
            }
            color.w *= pb.strength/(2.0 * ubo.params.max_pheromone_strength);

            fcolor = color;
        } else {
            fcolor = vec4(0.0);
        }
    }
#endif // RENDER_PHEROMONES_FRAG_PASS

#ifdef RENDER_ANTS_VERT_PASS
    layout(location = 0) out vec4 vcolor;
    layout(location = 1) out vec2 vuv;
    void main() {
        int ant_index = gl_VertexIndex / 6;
        int vert_index = gl_VertexIndex % 6;

        Ant p = ants[ant_index];
        AntType t = ant_types[p.type_index];
        vec2 vpos = quad_verts[vert_index].xy;

        float zoom = ubo.params.zoom;
        float ant_size = t.visual_radius * ubo.params.visual_radius_scale;
        vec2 mres = vec2(ubo.frame.monitor_width, ubo.frame.monitor_height);
        vec2 wres = vec2(ubo.frame.width, ubo.frame.height);

        vec2 pos = p.pos.xy + ubo.camera.eye.xy - vec2(float(ubo.params.world_size_x), float(ubo.params.world_size_y)) * 0.5;
        pos += vpos * 0.5 * ant_size;
        pos /= mres; // world space to 0..1
        pos *= mres/wres; // 0..1 scaled wrt window size
        pos *= zoom;
        pos *= 2.0;
        gl_Position = vec4(pos, 0.0, 1.0);

        vcolor = t.color;
        vuv = quad_uvs[vert_index];
    }
#endif // RENDER_ANTS_VERT_PASS

#ifdef RENDER_ANTS_FRAG_PASS
    layout(location = 0) in vec4 vcolor;
    layout(location = 1) in vec2 vuv;
    layout(location = 0) out vec4 fcolor;
    void main() {
        float zoom = ubo.params.zoom;
        float distanceFromCenter = length(vuv.xy - 0.5);
        float mask = 1.0 - smoothstep(0.5 - 0.5/zoom, 0.5, distanceFromCenter);
        // mask = pow(1.0 - distanceFromCenter, 4.5) * mask;
        // mask = 0;
        fcolor = vec4(vcolor.xyz, vcolor.a * mask);
    }
#endif // RENDER_ANTS_FRAG_PASS
