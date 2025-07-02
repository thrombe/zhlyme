 // This file is generated from code. DO NOT EDIT.

 struct DrawCall {
     uint index_count;
     uint instance_count;
     uint first_index;
     int vertex_offset;
     uint first_instance;
 };

 struct AntType {
     vec4 color;
     float pheromone_attraction;
     float pheromone_strength;
     float collision_radius;
     float visual_radius;
     float collision_strength;
 };

 struct Ant {
     vec2 pos;
     vec2 vel;
     uint type_index;
     float age;
     float exposure;
 };

 struct Params {
     float delta;
     uint grid_size;
     float zoom;
     float visual_radius_scale;
     uint randomize_ant_types;
     uint randomize_ant_attrs;
     uint ant_type_count;
     uint ant_count;
     uint spawn_count;
     int bin_size;
     int bin_buf_size;
     int bin_buf_size_x;
     int bin_buf_size_y;
     int world_size_x;
     int world_size_y;
     float entropy;
     float friction;
     int half_spread_max;
     int world_wrapping;
     float max_pheromone_strength;
     float collision_radius_scale;
     float collision_strength_scale;
 };

 struct PushConstantsCompute {
     int seed;
 };

 struct PushConstantsReduce {
     int seed;
     int step;
 };

 struct PushConstantsBlur {
     int seed;
     int dimension;
 };

 struct Camera2DMeta {
     uint did_move;
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

 const int _set_render = 0;
 const int _set_ant_bins = 1;
 const int _set_pheromones = 2;

 const int _bind_camera = 0;
 const int _bind_ants_draw_call = 1;
 const int _bind_scratch = 2;
 const int _bind_ant_types = 3;
 const int _bind_ants_back = 4;
 const int _bind_ants = 5;
 const int _bind_ant_bins_back = 6;
 const int _bind_ant_bins = 7;
 const int _bind_pheromones_back = 8;
 const int _bind_pheromones = 9;

