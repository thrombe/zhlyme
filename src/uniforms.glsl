 // This file is generated from code. DO NOT EDIT.

 struct DrawCall {
     uint index_count;
     uint instance_count;
     uint first_index;
     int vertex_offset;
     uint first_instance;
 };

 struct ParticleType {
     vec4 color;
     float particle_scale;
     uint _pad0;
     uint _pad1;
     uint _pad2;
 };

 struct ParticleForce {
     float attraction_strength;
     float attraction_radius;
     float attraction_peak_dist_factor;
     float collision_strength;
     float collision_radius;
     uint _pad0;
     uint _pad1;
     uint _pad2;
 };

 struct Particle {
     vec3 pos;
     vec3 vel;
     uint type_index;
     float age;
     float exposure;
     uint _pad0;
 };

 struct Params {
     float delta;
     uint particle_visual_size;
     uint grid_size;
     float zoom;
     float particle_z_shrinking_factor;
     float particle_z_blur_factor;
     float friction;
     float entropy;
     float collision_strength_scale;
     float attraction_strength_scale;
     float max_attraction_factor;
     uint randomize_particle_types;
     uint randomize_particle_attrs;
     uint particle_type_count;
     uint particle_count;
     uint spawn_count;
     int bin_size;
     int bin_buf_size;
     int bin_buf_size_x;
     int bin_buf_size_y;
     int bin_buf_size_z;
     int world_size_x;
     int world_size_y;
     int world_size_z;
 };

 struct PushConstants {
     int reduce_step;
     int seed;
     uint _pad0;
     uint _pad1;
 };

 struct Camera2DMeta {
     uint did_move;
     uint _pad1;
     uint _pad2;
     uint _pad3;
 };

 struct Camera2D {
     vec4 eye;
     Camera2DMeta meta;
 };

 struct Frame {
     uint frame;
     float time;
     float deltatime;
     int width;
     int height;
     int monitor_width;
     int monitor_height;
     uint pad0;
 };

 struct Mouse {
     int x;
     int y;
     uint left;
     uint right;
 };

 struct Uniforms {
     Camera2D camera;
     Frame frame;
     Mouse mouse;
     Params params;
 };

 const int _bind_camera = 0;
 const int _bind_particles_draw_call = 1;
 const int _bind_scratch = 2;
 const int _bind_particle_types = 3;
 const int _bind_particle_force_matrix = 4;
 const int _bind_particles_back = 5;
 const int _bind_particles = 6;
 const int _bind_particle_bins_back = 7;
 const int _bind_particle_bins = 8;

