const std = @import("std");

const vk = @import("vulkan");

const utils_mod = @import("utils.zig");
const Fuse = utils_mod.Fuse;
const ShaderUtils = utils_mod.ShaderUtils;
const Telemetry = utils_mod.Tracy;
const cast = utils_mod.cast;

const math = @import("math.zig");
const Vec4 = math.Vec4;
const Vec3 = math.Vec3;

const assets_mod = @import("assets.zig");

const engine_mod = @import("engine.zig");
const Engine = engine_mod.Engine;
const c = engine_mod.c;
const Device = engine_mod.VulkanContext.Api.Device;

const gui = @import("gui.zig");
const GuiEngine = gui.GuiEngine;

const render_utils = @import("render_utils.zig");
const Swapchain = render_utils.Swapchain;
const Buffer = render_utils.Buffer;
const Image = render_utils.Image;
const GraphicsPipeline = render_utils.GraphicsPipeline;
const ComputePipeline = render_utils.ComputePipeline;
const RenderPass = render_utils.RenderPass;
const DescriptorPool = render_utils.DescriptorPool;
const DescriptorSet = render_utils.DescriptorSet;
const CmdBuffer = render_utils.CmdBuffer;

const main = @import("main.zig");
const allocator = main.allocator;

pub const App = @This();

screen_image: Image,
depth_image: Image,
resources: ResourceManager,
descriptor_pool: DescriptorPool,
command_pool: vk.CommandPool,

telemetry: Telemetry,

pub fn init(engine: *Engine) !@This() {
    var ctx = &engine.graphics;
    const device = &ctx.device;

    const res = try engine.window.get_res();

    var telemetry = try utils_mod.Tracy.init();
    errdefer telemetry.deinit();

    const cmd_pool = try device.createCommandPool(&.{
        .queue_family_index = ctx.graphics_queue.family,
        .flags = .{
            .reset_command_buffer_bit = true,
        },
    }, null);
    errdefer device.destroyCommandPool(cmd_pool, null);

    var screen = try Image.new(ctx, cmd_pool, .{
        .img_type = .@"2d",
        .img_view_type = .@"2d",
        .format = .r16g16b16a16_sfloat,
        .layout = .color_attachment_optimal,
        .extent = .{
            .width = res.width,
            .height = res.height,
            .depth = 1,
        },
        .usage = .{
            .transfer_src_bit = true,
            .color_attachment_bit = true,
        },
        .view_aspect_mask = .{
            .color_bit = true,
        },
    });
    errdefer screen.deinit(device);

    var depth = try Image.new(ctx, cmd_pool, .{
        .img_type = .@"2d",
        .img_view_type = .@"2d",
        .format = .d32_sfloat,
        .layout = .depth_stencil_attachment_optimal,
        .extent = .{
            .width = res.width,
            .height = res.height,
            .depth = 1,
        },
        .usage = .{
            .depth_stencil_attachment_bit = true,
        },
        .view_aspect_mask = .{
            .depth_bit = true,
        },
    });
    errdefer depth.deinit(device);

    // TODO: pass params somehow or support dynamic params
    var resources = try ResourceManager.init(engine, cmd_pool, .{});
    errdefer resources.deinit(device);

    var desc_pool = try DescriptorPool.new(device);
    errdefer desc_pool.deinit(device);

    return @This(){
        .screen_image = screen,
        .depth_image = depth,
        .resources = resources,
        .descriptor_pool = desc_pool,
        .command_pool = cmd_pool,

        .telemetry = telemetry,
    };
}

pub fn deinit(self: *@This(), device: *Device) void {
    defer device.destroyCommandPool(self.command_pool, null);
    defer self.screen_image.deinit(device);
    defer self.depth_image.deinit(device);
    defer self.resources.deinit(device);
    defer self.descriptor_pool.deinit(device);

    defer self.telemetry.deinit();
}

pub fn pre_reload(self: *@This()) !void {
    _ = self;
}

pub fn post_reload(self: *@This()) !void {
    _ = self;
}

pub fn tick(
    self: *@This(),
    engine: *Engine,
    app_state: *AppState,
    gui_renderer: *GuiEngine.GuiRenderer,
    gui_state: *GuiState,
    renderer_state: *RendererState,
) !bool {
    self.telemetry.mark_frame() catch |e| utils_mod.dump_error(e);
    self.telemetry.begin_sample(@src(), "frame.tick");
    defer self.telemetry.end_sample();
    self.telemetry.plot("last frame time (ms)", app_state.ticker.real.delta * std.time.ms_per_s);

    const ctx = &engine.graphics;

    if (engine.window.should_close()) return false;

    if (engine.window.is_minimized()) {
        return true;
    }

    gui_renderer.render_start();

    try app_state.tick(engine, self);

    {
        self.telemetry.begin_sample(@src(), "gui_state.tick");
        defer self.telemetry.end_sample();

        try gui_state.tick(self, app_state);
    }
    {
        self.telemetry.begin_sample(@src(), "gui_renderer.render_end");
        defer self.telemetry.end_sample();

        try gui_renderer.render_end(&engine.graphics.device, &renderer_state.swapchain);
    }

    {
        self.telemetry.begin_sample(@src(), ".queue_wait_idle");
        defer self.telemetry.end_sample();

        // TODO: might be useful to create some kinda double buffered setup for
        //  cmdbuffers so that i can queue them before .queueWaitIdle()
        // multiple framebuffers => multiple descriptor sets => different buffers
        // big buffers that depends on the last frame's big buffer + multiple framebuffers => me sad
        // so just wait for one frame's queue to be empty before trying to render another frame
        try ctx.device.queueWaitIdle(ctx.graphics_queue.handle);
    }

    {
        self.telemetry.begin_sample(@src(), ".framerate_cap_sleep");
        defer self.telemetry.end_sample();

        const frametime = @as(f32, @floatFromInt(app_state.ticker.real.timer.read())) / std.time.ns_per_ms;
        const min_frametime = 1.0 / @as(f32, @floatFromInt(app_state.fps_cap)) * std.time.ms_per_s;
        if (frametime < min_frametime) {
            std.Thread.sleep(@intFromFloat(std.time.ns_per_ms * (min_frametime - frametime)));
        }
    }

    {
        self.telemetry.begin_sample(@src(), ".gpu_buffer_uploads");
        defer self.telemetry.end_sample();

        try self.resources.upload(&ctx.device);
    }

    if (renderer_state.stages.update()) {
        _ = app_state.shader_fuse.fuse();
    }

    if (app_state.shader_fuse.unfuse()) {
        self.telemetry.begin_sample(@src(), ".recreating_pipelins");
        defer self.telemetry.end_sample();
        try renderer_state.recreate_pipelines(engine, self, app_state);
    }

    if (app_state.cmdbuf_fuse.unfuse()) {
        self.telemetry.begin_sample(@src(), ".recreating_command_buffers");
        defer self.telemetry.end_sample();
        try renderer_state.recreate_cmdbuf(engine, self, app_state);
    }

    {
        self.telemetry.begin_sample(@src(), ".present");
        defer self.telemetry.end_sample();

        try renderer_state.swapchain.present_start(ctx);
        const present_state = renderer_state.swapchain.present_end(
            &[_]vk.CommandBuffer{
                renderer_state.cmdbuffer.bufs[renderer_state.swapchain.image_index],
                gui_renderer.cmd_bufs[renderer_state.swapchain.image_index],
            },
            ctx,
        ) catch |err| switch (err) {
            error.OutOfDateKHR => blk: {
                _ = app_state.resize_fuse.fuse();
                break :blk .suboptimal;
            },
            else => |narrow| return narrow,
        };
        // IDK: this never triggers :/
        if (present_state == .suboptimal) {
            std.debug.print("{any}\n", .{present_state});
        }
    }

    if (engine.window.resize_fuse.unfuse()) {
        _ = app_state.resize_fuse.fuse();
    }

    // this has to happen before the next app/gui tick
    if (app_state.resize_fuse.unfuse()) {
        self.telemetry.begin_sample(@src(), ".recreating_swapchain");
        defer self.telemetry.end_sample();
        // this is not good :/
        // we have to wait for queue to be idle before creating swapchain again
        try ctx.device.queueWaitIdle(ctx.graphics_queue.handle);

        try renderer_state.recreate_swapchain(engine, app_state);

        gui_renderer.deinit(&engine.graphics.device);
        gui_renderer.* = try GuiEngine.GuiRenderer.init(engine, &renderer_state.swapchain);
    }

    return true;
}

pub const ResourceManager = struct {
    uniform: Uniforms,
    uniform_buf: Buffer,

    particle_types: []ParticleType,
    particle_force_matrix: []ParticleForce,

    scratch_buf: Buffer,
    particle_types_buf: Buffer,
    particle_force_matrix_buf: Buffer,
    particles_back_buf: Buffer,
    particles_buf: Buffer,
    particle_bins_back_buf: Buffer,
    particle_bins_buf: Buffer,
    // updated from gpu side
    particles_draw_call_buf: Buffer,

    pub fn init(engine: *Engine, pool: vk.CommandPool, v: struct {
        num_particles: u32 = 100000,
        particle_type_count: u32 = 10,
    }) !@This() {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        var uniform_buf = try Buffer.new_initialized(ctx, .{
            .size = @sizeOf(Uniforms),
            .usage = .{ .uniform_buffer_bit = true },
            .memory_type = .{
                // https://community.khronos.org/t/memory-type-practice-for-an-mvp-uniform-buffer/109458/7
                // we ideally want device local for cpu to gpu, but instance transforms are not a bottleneck (generally)
                // so we save this memory (device_local + host_visible) for more useful things
                // .device_local_bit = true,

                .host_visible_bit = true,
                .host_coherent_bit = true,
            },
            .desc_type = .uniform_buffer,
        }, std.mem.zeroes(Uniforms), pool);
        errdefer uniform_buf.deinit(device);

        const particle_types = try allocator.alloc(ParticleType, v.particle_type_count);
        errdefer allocator.free(particle_types);
        @memset(particle_types, std.mem.zeroes(ParticleType));

        const particle_force_matrix = try allocator.alloc(ParticleForce, v.particle_type_count * v.particle_type_count);
        errdefer allocator.free(particle_force_matrix);
        @memset(particle_force_matrix, std.mem.zeroes(ParticleForce));

        var particle_types_buf = try Buffer.new_from_slice(ctx, .{
            .usage = .{ .storage_buffer_bit = true },
            .memory_type = .{ .device_local_bit = true, .host_visible_bit = true, .host_coherent_bit = true },
        }, particle_types, pool);
        errdefer particle_types_buf.deinit(device);

        var particle_force_matrix_buf = try Buffer.new_from_slice(ctx, .{
            .usage = .{ .storage_buffer_bit = true },
            .memory_type = .{ .device_local_bit = true, .host_visible_bit = true, .host_coherent_bit = true },
        }, particle_force_matrix, pool);
        errdefer particle_force_matrix_buf.deinit(device);

        var particles_back = try Buffer.new(ctx, .{
            .size = @sizeOf(Particle) * v.num_particles,
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer particles_back.deinit(device);

        var particles = try Buffer.new(ctx, .{
            .size = @sizeOf(Particle) * v.num_particles,
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer particles.deinit(device);

        // TODO: better bin buffer sizes
        const res = .{ .width = @as(u32, 1000), .height = @as(u32, 1000) };
        // 1 larger than the max number of bins
        var particle_bins_back = try Buffer.new(ctx, .{
            .size = @sizeOf(i32) * (1 + res.width * res.height) * 5,
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer particle_bins_back.deinit(device);

        var particle_bins = try Buffer.new(ctx, .{
            .size = @sizeOf(i32) * (1 + res.width * res.height),
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer particle_bins.deinit(device);

        var scratch = try Buffer.new(ctx, .{
            .size = 4 * 4 * 100,
            .usage = .{ .storage_buffer_bit = true },
        });
        errdefer scratch.deinit(device);

        var draw_call = try Buffer.new_initialized(ctx, .{
            .size = 1,
            .usage = .{ .storage_buffer_bit = true, .indirect_buffer_bit = true },
        }, std.mem.zeroes(DrawCall), pool);
        errdefer draw_call.deinit(device);

        return @This(){
            .uniform = std.mem.zeroes(Uniforms),
            .uniform_buf = uniform_buf,
            .particles_draw_call_buf = draw_call,
            .scratch_buf = scratch,
            .particle_types = particle_types,
            .particle_force_matrix = particle_force_matrix,
            .particle_types_buf = particle_types_buf,
            .particle_force_matrix_buf = particle_force_matrix_buf,
            .particles_back_buf = particles_back,
            .particles_buf = particles,
            .particle_bins_back_buf = particle_bins_back,
            .particle_bins_buf = particle_bins,
        };
    }

    pub fn deinit(self: *@This(), device: *Device) void {
        self.uniform_buf.deinit(device);
        self.particles_draw_call_buf.deinit(device);
        self.scratch_buf.deinit(device);
        allocator.free(self.particle_types);
        self.particle_types_buf.deinit(device);
        allocator.free(self.particle_force_matrix);
        self.particle_force_matrix_buf.deinit(device);
        self.particles_back_buf.deinit(device);
        self.particles_buf.deinit(device);
        self.particle_bins_back_buf.deinit(device);
        self.particle_bins_buf.deinit(device);
    }

    pub fn add_binds(self: *@This(), builder: *render_utils.DescriptorSet.Builder, builder_back: *render_utils.DescriptorSet.Builder) !void {
        const add_to_set = struct {
            fn func(set_builder: @TypeOf(builder), buf: *Buffer, bind: UniformBinds) !void {
                try set_builder.add(buf, bind.bind());
            }
        }.func;

        try add_to_set(builder, &self.uniform_buf, .camera);
        try add_to_set(builder, &self.particles_draw_call_buf, .particles_draw_call);
        try add_to_set(builder, &self.scratch_buf, .scratch);
        try add_to_set(builder, &self.particle_types_buf, .particle_types);
        try add_to_set(builder, &self.particle_force_matrix_buf, .particle_force_matrix);
        try add_to_set(builder, &self.particles_back_buf, .particles_back);
        try add_to_set(builder, &self.particles_buf, .particles);
        try add_to_set(builder, &self.particle_bins_back_buf, .particle_bins_back);
        try add_to_set(builder, &self.particle_bins_buf, .particle_bins);

        try add_to_set(builder_back, &self.uniform_buf, .camera);
        try add_to_set(builder_back, &self.particles_draw_call_buf, .particles_draw_call);
        try add_to_set(builder_back, &self.scratch_buf, .scratch);
        try add_to_set(builder_back, &self.particle_types_buf, .particle_types);
        try add_to_set(builder_back, &self.particle_force_matrix_buf, .particle_force_matrix);
        try add_to_set(builder_back, &self.particles_back_buf, .particles_back);
        try add_to_set(builder_back, &self.particles_buf, .particles);
        try add_to_set(builder_back, &self.particle_bins_buf, .particle_bins_back);
        try add_to_set(builder_back, &self.particle_bins_back_buf, .particle_bins);
    }

    pub fn upload(self: *@This(), device: *Device) !void {
        try self.update_uniforms(device);
        try self.update_particle_types(device);
        try self.update_force_matrix(device);
    }

    fn update_uniforms(self: *@This(), device: *Device) !void {
        const maybe_mapped = try device.mapMemory(self.uniform_buf.memory, 0, vk.WHOLE_SIZE, .{});
        const mapped = maybe_mapped orelse return error.MappingMemoryFailed;
        defer device.unmapMemory(self.uniform_buf.memory);

        const mem: *Uniforms = @ptrCast(@alignCast(mapped));
        mem.* = self.uniform;
    }

    fn update_particle_types(self: *@This(), device: *Device) !void {
        const maybe_mapped = try device.mapMemory(self.particle_types_buf.memory, 0, vk.WHOLE_SIZE, .{});
        const mapped = maybe_mapped orelse return error.MappingMemoryFailed;
        defer device.unmapMemory(self.particle_types_buf.memory);

        const buf = self.particle_types;
        const mem: [*c]ParticleType = @ptrCast(@alignCast(mapped));
        @memcpy(mem[0..buf.len], buf);
    }

    fn update_force_matrix(self: *@This(), device: *Device) !void {
        const maybe_mapped = try device.mapMemory(self.particle_force_matrix_buf.memory, 0, vk.WHOLE_SIZE, .{});
        const mapped = maybe_mapped orelse return error.MappingMemoryFailed;
        defer device.unmapMemory(self.particle_force_matrix_buf.memory);

        const buf = self.particle_force_matrix;
        const mem: [*c]ParticleForce = @ptrCast(@alignCast(mapped));
        @memcpy(mem[0..buf.len], buf);
    }

    pub const UniformBinds = enum(u32) {
        camera,
        particles_draw_call,
        scratch,
        particle_types,
        particle_force_matrix,
        particles_back,
        particles,
        particle_bins_back,
        particle_bins,

        pub fn bind(self: @This()) u32 {
            return @intFromEnum(self);
        }
    };
    pub const ParticleType = extern struct {
        color: Vec4,
        particle_scale: f32,

        _pad0: u32 = 0,
        _pad1: u32 = 0,
        _pad2: u32 = 0,
    };
    pub const ParticleForce = extern struct {
        attraction_strength: f32,
        attraction_radius: f32,
        attraction_peak_dist_factor: f32,
        collision_strength: f32,
        collision_radius: f32,

        _pad0: u32 = 0,
        _pad1: u32 = 0,
        _pad2: u32 = 0,
    };
    pub const Particle = extern struct {
        pos: math.Vec3,
        vel: math.Vec3,
        type_index: u32,
        age: f32,
        exposure: f32,

        _pad0: u32 = 0,
        // _pad1: u32 = 0,
        // _pad2: u32 = 0,
    };
    pub const DrawCall = vk.DrawIndexedIndirectCommand;

    pub const PushConstants = extern struct {
        reduce_step: i32 = 0,
        seed: i32,

        _pad0: u32 = 0,
        _pad1: u32 = 0,
        // _pad2: u32 = 0,
    };

    pub const Uniforms = extern struct {
        camera: utils_mod.ShaderUtils.Camera2D,
        frame: utils_mod.ShaderUtils.Frame,
        mouse: utils_mod.ShaderUtils.Mouse,
        params: Params,

        const Params = extern struct {
            delta: f32 = 0,
            particle_visual_size: u32 = 16,
            grid_size: u32 = 32,
            zoom: f32 = 1.0,
            particle_z_shrinking_factor: f32 = 0.7,
            particle_z_blur_factor: f32 = 0.27,
            friction: f32,
            entropy: f32 = 0.0,
            collision_strength_scale: f32 = 96,
            attraction_strength_scale: f32 = 27,
            max_attraction_factor: f32 = 27,
            randomize_particle_types: u32 = 0,
            randomize_particle_attrs: u32 = 0,
            particle_type_count: u32 = 0,
            particle_count: u32 = 0,
            spawn_count: u32,
            bin_size: i32,
            bin_buf_size: i32,
            bin_buf_size_x: i32,
            bin_buf_size_y: i32,
            bin_buf_size_z: i32,
            world_size_x: i32,
            world_size_y: i32,
            world_size_z: i32,

            // _pad0: u32 = 0,
            // _pad1: u32 = 0,
            // _pad2: u32 = 0,
        };

        fn from(
            state: *AppState,
            window: *engine_mod.Window,
        ) !@This() {
            // const inputs = window.input();

            state.params.delta = state.ticker.scaled.delta / @as(f32, @floatFromInt(state.steps_per_frame));

            const spawn_count = @min(state.spawn_count, 64);
            state.spawn_count -= spawn_count;

            const particle_count = state.params.particle_count;
            state.params.particle_count = @min(particle_count + spawn_count, state.max_particle_count);
            state.params.spawn_count = state.params.particle_count - particle_count;
            state.params.friction = @exp(-state.friction * state.params.delta);
            state.params.particle_type_count = state.particle_type_count;
            state.params.randomize_particle_types = @intCast(@intFromBool(state.randomize.particle_types));
            state.params.randomize_particle_attrs = @intCast(@intFromBool(state.randomize.particle_attrs));

            // cap requested world size's z coord.
            state.requested_world_size.z = @min(state.requested_world_size.z, state.bin_size * state.bin_buf_size_z_max);

            state.params.bin_size = state.bin_size;
            state.params.bin_buf_size_x = @divFloor(state.requested_world_size.x, state.bin_size);
            state.params.bin_buf_size_y = @divFloor(state.requested_world_size.y, state.bin_size);
            state.params.bin_buf_size_z = @divFloor(state.requested_world_size.z, state.bin_size);

            // allow z to be 0
            state.params.world_size_x = state.params.bin_buf_size_x * state.bin_size;
            state.params.world_size_y = state.params.bin_buf_size_y * state.bin_size;
            state.params.world_size_z = state.params.bin_buf_size_z * state.bin_size;

            if (state.params.world_size_x == 0) {
                state.params.world_size_x = state.requested_world_size.x;
            }
            if (state.params.world_size_y == 0) {
                state.params.world_size_y = state.requested_world_size.y;
            }
            if (state.params.world_size_z == 0) {
                state.params.world_size_z = state.requested_world_size.z;
            }

            // bin_buf_size_. > 0
            state.params.bin_buf_size_z = @max(state.params.bin_buf_size_z, 1);
            state.params.bin_buf_size = state.params.bin_buf_size_x * state.params.bin_buf_size_y * state.params.bin_buf_size_z;

            // TODO: don't fuse every frame man
            _ = state.cmdbuf_fuse.fuse();

            if (spawn_count > 0) _ = state.cmdbuf_fuse.fuse();

            const uniform = @This(){
                .camera = state.camera,
                .mouse = .{
                    .x = state.mouse.x,
                    .y = state.mouse.y,
                    .left = @intCast(@intFromBool(state.mouse.left)),
                    .right = @intCast(@intFromBool(state.mouse.right)),
                },
                .frame = .{
                    .frame = state.frame,
                    .time = state.ticker.scaled.time_f,
                    .deltatime = state.ticker.scaled.delta,
                    .width = @intCast(window.extent.width),
                    .height = @intCast(window.extent.height),
                    .monitor_width = @intCast(state.monitor_rez.width),
                    .monitor_height = @intCast(state.monitor_rez.height),
                },
                .params = state.params,
            };

            return uniform;
        }
    };
};

pub const RendererState = struct {
    swapchain: Swapchain,
    cmdbuffer: CmdBuffer,
    descriptor_set: DescriptorSet,
    descriptor_set_back: DescriptorSet,

    stages: ShaderStageManager,
    pipelines: Pipelines,

    // not owned
    pool: vk.CommandPool,

    const Pipelines = struct {
        bg: GraphicsPipeline,
        render: GraphicsPipeline,
        spawn_particles: ComputePipeline,
        bin_reset: ComputePipeline,
        particle_count: ComputePipeline,
        bin_prefix_sum: ComputePipeline,
        particle_binning: ComputePipeline,
        tick_particles: ComputePipeline,

        fn deinit(self: *@This(), device: *Device) void {
            self.bg.deinit(device);
            self.render.deinit(device);
            self.spawn_particles.deinit(device);
            self.bin_reset.deinit(device);
            self.particle_count.deinit(device);
            self.bin_prefix_sum.deinit(device);
            self.particle_binning.deinit(device);
            self.tick_particles.deinit(device);
        }
    };

    pub fn init(app: *App, engine: *Engine, app_state: *AppState) !@This() {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        var arena = std.heap.ArenaAllocator.init(allocator.*);
        defer arena.deinit();
        const alloc = arena.allocator();

        var gen = try utils_mod.ShaderUtils.GlslBindingGenerator.init();
        defer gen.deinit();
        try gen.add_struct("DrawCall", ResourceManager.DrawCall);
        try gen.add_struct("ParticleType", ResourceManager.ParticleType);
        try gen.add_struct("ParticleForce", ResourceManager.ParticleForce);
        try gen.add_struct("Particle", ResourceManager.Particle);
        try gen.add_struct("Params", ResourceManager.Uniforms.Params);
        try gen.add_struct("PushConstants", ResourceManager.PushConstants);
        try gen.add_struct("Uniforms", ResourceManager.Uniforms);
        try gen.add_enum("_bind", ResourceManager.UniformBinds);
        try gen.dump_shader("src/uniforms.glsl");

        var shader_stages = std.ArrayList(utils_mod.ShaderCompiler.ShaderInfo).init(alloc);
        const includes = try alloc.dupe([]const u8, &[_][]const u8{"src"});
        try shader_stages.append(.{
            .name = "bg_frag",
            .stage = .fragment,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{"BG_FRAG_PASS"}),
        });
        try shader_stages.append(.{
            .name = "bg_vert",
            .stage = .vertex,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{"BG_VERT_PASS"}),
        });
        try shader_stages.append(.{
            .name = "render_frag",
            .stage = .fragment,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{"RENDER_FRAG_PASS"}),
        });
        try shader_stages.append(.{
            .name = "render_vert",
            .stage = .vertex,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{"RENDER_VERT_PASS"}),
        });
        try shader_stages.append(.{
            .name = "spawn_particles",
            .stage = .compute,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{ "SPAWN_PARTICLES_PASS", "COMPUTE_PASS" }),
        });
        try shader_stages.append(.{
            .name = "bin_reset",
            .stage = .compute,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{ "BIN_RESET_PASS", "COMPUTE_PASS" }),
        });
        try shader_stages.append(.{
            .name = "particle_count",
            .stage = .compute,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{ "PARTICLE_COUNT_PASS", "COMPUTE_PASS" }),
        });
        try shader_stages.append(.{
            .name = "bin_prefix_sum",
            .stage = .compute,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{ "BIN_PREFIX_SUM_PASS", "COMPUTE_PASS" }),
        });
        try shader_stages.append(.{
            .name = "particle_binning",
            .stage = .compute,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{ "PARTICLE_BINNING_PASS", "COMPUTE_PASS" }),
        });
        try shader_stages.append(.{
            .name = "tick_particles",
            .stage = .compute,
            .path = "src/shader.glsl",
            .include = includes,
            .define = try alloc.dupe([]const u8, &[_][]const u8{ "TICK_PARTICLES_PASS", "COMPUTE_PASS" }),
        });

        var stages = try ShaderStageManager.init(shader_stages.items);
        errdefer stages.deinit();

        var swapchain = try Swapchain.init(ctx, engine.window.extent, .{
            // .prefer_present_mode = .immediate_khr,
        });
        errdefer swapchain.deinit(device);

        var self: @This() = .{
            .stages = stages,
            .pipelines = undefined,
            .descriptor_set = undefined,
            .descriptor_set_back = undefined,
            .swapchain = swapchain,
            .pool = app.command_pool,
            .cmdbuffer = undefined,
        };

        try self.create_pipelines(engine, app, false);
        errdefer self.descriptor_set.deinit(device);
        errdefer self.descriptor_set_back.deinit(device);
        errdefer self.pipelines.deinit(device);

        self.cmdbuffer = try self.create_cmdbuf(engine, app, app_state);
        errdefer self.cmdbuffer.deinit(device);

        return self;
    }

    pub fn recreate_pipelines(self: *@This(), engine: *Engine, app: *App, app_state: *AppState) !void {
        try self.create_pipelines(engine, app, true);
        _ = app_state.cmdbuf_fuse.fuse();
    }

    pub fn recreate_swapchain(self: *@This(), engine: *Engine, app_state: *AppState) !void {
        try self.swapchain.recreate(&engine.graphics, engine.window.extent, .{});
        _ = app_state.cmdbuf_fuse.fuse();
    }

    pub fn recreate_cmdbuf(self: *@This(), engine: *Engine, app: *App, app_state: *AppState) !void {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        const cmdbuffer = try self.create_cmdbuf(engine, app, app_state);
        self.cmdbuffer.deinit(device);
        self.cmdbuffer = cmdbuffer;
    }

    fn create_pipelines(self: *@This(), engine: *Engine, app: *App, initialized: bool) !void {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        var desc_set_builder = app.descriptor_pool.set_builder();
        defer desc_set_builder.deinit();
        var desc_set_builder_back = app.descriptor_pool.set_builder();
        defer desc_set_builder_back.deinit();

        try app.resources.add_binds(&desc_set_builder, &desc_set_builder_back);

        var desc_set = try desc_set_builder.build(device);
        errdefer desc_set.deinit(device);
        var desc_set_back = try desc_set_builder_back.build(device);
        errdefer desc_set_back.deinit(device);

        if (initialized) {
            self.pipelines.bg.deinit(device);
        }
        self.pipelines.bg = try GraphicsPipeline.new(device, .{
            .vert = self.stages.shaders.map.get("bg_vert").?.code,
            .frag = self.stages.shaders.map.get("bg_frag").?.code,
            .dynamic_info = .{
                .image_format = app.screen_image.format,
                .depth_format = app.depth_image.format,
            },
            .desc_set_layouts = &.{desc_set.layout},
            .cull_mode = .{},
            .render_mode = .solid_triangles,
        });

        if (initialized) {
            self.pipelines.render.deinit(device);
        }
        self.pipelines.render = try GraphicsPipeline.new(device, .{
            .vert = self.stages.shaders.map.get("render_vert").?.code,
            .frag = self.stages.shaders.map.get("render_frag").?.code,
            .dynamic_info = .{
                .image_format = app.screen_image.format,
                .depth_format = app.depth_image.format,
            },
            .desc_set_layouts = &.{
                desc_set.layout,
            },
            .cull_mode = .{},
            .render_mode = .solid_triangles,
            .alpha_blend = .{
                .blend_enable = vk.TRUE,
                .src_color_blend_factor = .src_alpha,
                .dst_color_blend_factor = .one,
                .color_blend_op = .add,
                .src_alpha_blend_factor = .one,
                .dst_alpha_blend_factor = .zero,
                .alpha_blend_op = .add,
                .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            },
        });

        if (initialized) {
            self.pipelines.spawn_particles.deinit(device);
        }
        self.pipelines.spawn_particles = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("spawn_particles").?.code,
            .desc_set_layouts = &.{desc_set.layout},
            .push_constant_ranges = &[_]vk.PushConstantRange{.{
                .stage_flags = .{ .compute_bit = true },
                .offset = 0,
                .size = @sizeOf(ResourceManager.PushConstants),
            }},
        });

        if (initialized) {
            self.pipelines.bin_reset.deinit(device);
        }
        self.pipelines.bin_reset = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("bin_reset").?.code,
            .desc_set_layouts = &.{desc_set.layout},
        });

        if (initialized) {
            self.pipelines.particle_count.deinit(device);
        }
        self.pipelines.particle_count = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("particle_count").?.code,
            .desc_set_layouts = &.{desc_set.layout},
        });

        if (initialized) {
            self.pipelines.bin_prefix_sum.deinit(device);
        }
        self.pipelines.bin_prefix_sum = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("bin_prefix_sum").?.code,
            .desc_set_layouts = &.{desc_set.layout},
            .push_constant_ranges = &[_]vk.PushConstantRange{.{
                .stage_flags = .{ .compute_bit = true },
                .offset = 0,
                .size = @sizeOf(ResourceManager.PushConstants),
            }},
        });

        if (initialized) {
            self.pipelines.particle_binning.deinit(device);
        }
        self.pipelines.particle_binning = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("particle_binning").?.code,
            .desc_set_layouts = &.{desc_set.layout},
            .push_constant_ranges = &[_]vk.PushConstantRange{.{
                .stage_flags = .{ .compute_bit = true },
                .offset = 0,
                .size = @sizeOf(ResourceManager.PushConstants),
            }},
        });

        if (initialized) {
            self.pipelines.tick_particles.deinit(device);
        }
        self.pipelines.tick_particles = try ComputePipeline.new(device, .{
            .shader = self.stages.shaders.map.get("tick_particles").?.code,
            .desc_set_layouts = &.{desc_set.layout},
            .push_constant_ranges = &[_]vk.PushConstantRange{.{
                .stage_flags = .{ .compute_bit = true },
                .offset = 0,
                .size = @sizeOf(ResourceManager.PushConstants),
            }},
        });

        if (initialized) {
            self.descriptor_set.deinit(device);
            self.descriptor_set_back.deinit(device);
        }
        self.descriptor_set = desc_set;
        self.descriptor_set_back = desc_set_back;
    }

    pub fn create_cmdbuf(self: *@This(), engine: *Engine, app: *App, app_state: *AppState) !CmdBuffer {
        const ctx = &engine.graphics;
        const device = &ctx.device;

        const alloc = app_state.arena.allocator();
        // _ = alloc;

        // std.mem.swap(DescriptorSet, &self.descriptor_set, &self.descriptor_set_back);

        var cmdbuf = try CmdBuffer.init(device, .{
            .pool = app.command_pool,
            .size = self.swapchain.swap_images.len,
        });
        errdefer cmdbuf.deinit(device);

        try cmdbuf.begin(device);

        // spawn particles
        cmdbuf.bindCompute(device, .{
            .pipeline = self.pipelines.spawn_particles,
            .desc_set = self.descriptor_set.set,
        });

        // TODO: oof. don't use arena allocator. somehow retain this memory somewhere.
        {
            const constants = try alloc.create(ResourceManager.PushConstants);
            constants.* = .{ .seed = app_state.rng.random().int(i32) };
            cmdbuf.push_constants(device, self.pipelines.spawn_particles.layout, std.mem.asBytes(constants), .{ .compute_bit = true });
        }
        cmdbuf.dispatch(device, .{ .x = 1 });
        cmdbuf.memBarrier(device, .{});

        for (0..app_state.steps_per_frame) |_| {
            // bin reset
            cmdbuf.bindCompute(device, .{
                .pipeline = self.pipelines.bin_reset,
                .desc_set = self.descriptor_set.set,
            });
            cmdbuf.dispatch(device, .{ .x = math.divide_roof(cast(u32, app_state.params.bin_buf_size), 64) });
            cmdbuf.memBarrier(device, .{});

            // count particles
            cmdbuf.bindCompute(device, .{
                .pipeline = self.pipelines.particle_count,
                .desc_set = self.descriptor_set.set,
            });
            cmdbuf.dispatch(device, .{ .x = math.divide_roof(app_state.params.particle_count, 64) });
            cmdbuf.memBarrier(device, .{});

            // bin count prefix sum
            var reduce_step: u5 = 0;
            while (true) : (reduce_step += 1) {
                cmdbuf.bindCompute(device, .{
                    .pipeline = self.pipelines.bin_prefix_sum,
                    .desc_set = self.descriptor_set.set,
                });

                {
                    const constants = try alloc.create(ResourceManager.PushConstants);
                    constants.* = .{ .reduce_step = reduce_step, .seed = app_state.rng.random().int(i32) };
                    cmdbuf.push_constants(device, self.pipelines.bin_prefix_sum.layout, std.mem.asBytes(constants), .{ .compute_bit = true });
                }

                // 1 larger then the buffer to store capacities
                cmdbuf.dispatch(device, .{ .x = math.divide_roof(cast(u32, app_state.params.bin_buf_size + 1), 64) });
                cmdbuf.memBarrier(device, .{});

                // std.debug.print("{any} {any}\n", .{ reduce_step, app_state.params.bin_buf_size - (@as(i32, 1) << reduce_step) });
                if (app_state.params.bin_buf_size - (@as(i32, 1) << reduce_step) < 0) {
                    break;
                }

                std.mem.swap(DescriptorSet, &self.descriptor_set, &self.descriptor_set_back);
            }

            // bin particles
            cmdbuf.bindCompute(device, .{
                .pipeline = self.pipelines.particle_binning,
                .desc_set = self.descriptor_set.set,
            });
            {
                const constants = try alloc.create(ResourceManager.PushConstants);
                constants.* = .{ .seed = app_state.rng.random().int(i32) };
                cmdbuf.push_constants(device, self.pipelines.particle_binning.layout, std.mem.asBytes(constants), .{ .compute_bit = true });
            }
            cmdbuf.dispatch(device, .{ .x = math.divide_roof(app_state.params.particle_count, 64) });
            cmdbuf.memBarrier(device, .{});

            // tick particles
            cmdbuf.bindCompute(device, .{
                .pipeline = self.pipelines.tick_particles,
                .desc_set = self.descriptor_set.set,
            });
            {
                const constants = try alloc.create(ResourceManager.PushConstants);
                constants.* = .{ .seed = app_state.rng.random().int(i32) };
                cmdbuf.push_constants(device, self.pipelines.tick_particles.layout, std.mem.asBytes(constants), .{ .compute_bit = true });
            }
            cmdbuf.dispatch(device, .{ .x = math.divide_roof(app_state.params.particle_count, 64) });
            cmdbuf.memBarrier(device, .{});
        }

        cmdbuf.dynamic_render_begin(device, .{
            .image = app.screen_image.view,
            .depth = app.depth_image.view,
            .extent = engine.window.extent,
        });

        // bg pass
        for (cmdbuf.bufs) |buf| {
            device.cmdBindPipeline(buf, .graphics, self.pipelines.bg.pipeline);
            device.cmdBindDescriptorSets(buf, .graphics, self.pipelines.bg.layout, 0, 1, @ptrCast(&self.descriptor_set.set), 0, null);
            device.cmdDraw(buf, 6, 1, 0, 0);
        }

        // render particles
        cmdbuf.draw_indirect(device, .{
            .pipeline = &self.pipelines.render,
            .desc_sets = &.{
                self.descriptor_set.set,
            },
            .calls = .{
                .buffer = app.resources.particles_draw_call_buf.buffer,
                .count = 1,
                .stride = @sizeOf(ResourceManager.DrawCall),
            },
        });

        cmdbuf.dynamic_render_end(device);
        cmdbuf.draw_into_swapchain(device, .{
            .image = app.screen_image.image,
            .image_layout = .color_attachment_optimal,
            .size = self.swapchain.extent,
            .swapchain = &self.swapchain,
            .queue_family = ctx.graphics_queue.family,
        });
        try cmdbuf.end(device);

        return cmdbuf;
    }

    pub fn deinit(self: *@This(), device: *Device) void {
        try self.swapchain.waitForAll(device);

        defer self.swapchain.deinit(device);
        defer self.cmdbuffer.deinit(device);

        defer self.descriptor_set.deinit(device);
        defer self.descriptor_set_back.deinit(device);

        defer self.stages.deinit();
        defer self.pipelines.deinit(device);
    }
};

const ShaderStageManager = struct {
    shaders: utils_mod.ShaderCompiler.Stages,
    compiler: utils_mod.ShaderCompiler.Compiler,

    pub fn init(stages: []const utils_mod.ShaderCompiler.ShaderInfo) !@This() {
        var comp = try utils_mod.ShaderCompiler.Compiler.init(.{ .opt = .fast, .env = .vulkan1_3 }, stages);
        errdefer comp.deinit();

        return .{
            .shaders = try utils_mod.ShaderCompiler.Stages.init(&comp, stages),
            .compiler = comp,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.shaders.deinit();
        self.compiler.deinit();
    }

    pub fn update(self: *@This()) bool {
        return self.shaders.update(&self.compiler);
    }
};

pub const AppState = struct {
    ticker: utils_mod.SimulationTicker,

    monitor_rez: struct { width: u32, height: u32 },
    mouse: extern struct { x: i32 = 0, y: i32 = 0, left: bool = false, right: bool = false } = .{},

    frame: u32 = 0,
    fps_cap: u32 = 60,

    rng: std.Random.Xoshiro256,
    resize_fuse: Fuse = .{},
    cmdbuf_fuse: Fuse = .{},
    shader_fuse: Fuse = .{},
    focus: bool = false,

    randomize: struct {
        particle_colors: bool = false,
        particle_forces: bool = false,
        particle_types: bool = false,
        particle_attrs: bool = false,
    } = .{},
    steps_per_frame: u32 = 2,
    max_particle_count: u32 = 100000,
    max_particle_type_count: u32 = 10,
    particle_type_count: u32 = 5,
    spawn_count: u32 = 15000,
    friction: f32 = 2.0,
    bin_size: i32 = 62,
    bin_buf_size_z_max: i32 = 5,
    requested_world_size: math.Vec3T(i32) = .{ .x = 1800, .y = 1200, .z = 13 },
    params: ResourceManager.Uniforms.Params = .{
        .spawn_count = 0,
        .friction = 0,

        .bin_size = 0,
        .bin_buf_size = 0,
        .bin_buf_size_x = 0,
        .bin_buf_size_y = 0,
        .bin_buf_size_z = 0,

        .world_size_x = 0,
        .world_size_y = 0,
        .world_size_z = 0,
    },
    camera: ShaderUtils.Camera2D = .{ .eye = .{} },

    arena: std.heap.ArenaAllocator,

    // fn interpolated(self: *const @This(), lt: *const C.LastTransform, t: *const C.GlobalTransform) C.Transform {
    //     return lt.transform.lerp(&t.transform, self.ticker.simulation.interpolation_factor);
    // }

    pub fn init(window: *engine_mod.Window, app: *App) !@This() {
        const mouse = window.poll_mouse();
        const sze = try window.get_res();

        var this = @This(){
            .ticker = try .init(),
            .monitor_rez = .{ .width = sze.width, .height = sze.height },
            .mouse = .{ .x = mouse.x, .y = mouse.y, .left = mouse.left },
            .rng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp())),
            .arena = std.heap.ArenaAllocator.init(allocator.*),
        };

        this.randomize_particle_forces(app);
        this.randomize_particle_colors(app);
        this.randomize_particle_types(app);
        this.randomize_particle_attrs(app);

        return this;
    }

    pub fn deinit(self: *@This()) void {
        self.arena.deinit();
    }

    pub fn pre_reload(self: *@This()) !void {
        _ = self;
    }

    pub fn post_reload(self: *@This()) !void {
        _ = self.resize_fuse.fuse();
        _ = self.shader_fuse.fuse();
        _ = self.cmdbuf_fuse.fuse();
    }

    fn randomize_particle_forces(self: *@This(), app: *App) void {
        const zrng = .{
            .collision_strength = math.Rng.init(self.rng.random()).with2(.{ .min = 0.7, .max = 1 }),
            .collision_radius = math.Rng.init(self.rng.random()).with2(.{ .min = 0.3, .max = 0.4 }),
            .attraction_strength = math.Rng.init(self.rng.random()).with2(.{ .min = -0.3, .max = 1.0 }),
            .attraction_radius = math.Rng.init(self.rng.random()).with2(.{ .min = 0.7, .max = 1 }),
            .attraction_peak_dist_factor = math.Rng.init(self.rng.random()).with2(.{ .min = 0.6, .max = 0.7 }),
        };
        for (app.resources.particle_force_matrix) |*pf| {
            pf.* = std.mem.zeroes(@TypeOf(pf.*));
            inline for (@typeInfo(@TypeOf(pf.*)).@"struct".fields) |f| {
                if (comptime @hasField(@TypeOf(zrng), f.name)) {
                    @field(pf, f.name) = @field(zrng, f.name).next();
                }
            }
        }

        self.randomize.particle_forces = true;
    }

    fn randomize_particle_colors(self: *@This(), app: *App) void {
        const zrng = .{
            .color = math.Rng.init(self.rng.random()).with2(.{ .min = 0.1, .max = 1.0 }),
            .particle_scale = math.Rng.init(self.rng.random()).with2(.{ .min = 0.4, .max = 0.6 }),
        };

        for (app.resources.particle_types) |*pt| {
            pt.* = .{
                .color = Vec3.random(&zrng.color).normalize().withw(1.0),
                .particle_scale = zrng.particle_scale.next(),
            };
        }

        self.randomize.particle_colors = true;
    }

    fn randomize_particle_types(self: *@This(), app: *App) void {
        _ = app;
        self.randomize.particle_types = true;
    }

    fn randomize_particle_attrs(self: *@This(), app: *App) void {
        _ = app;
        self.randomize.particle_attrs = true;
    }

    pub fn tick(self: *@This(), engine: *Engine, app: *App) !void {
        app.telemetry.begin_sample(@src(), "app_state.tick");
        defer app.telemetry.end_sample();

        defer _ = self.arena.reset(.retain_capacity);

        defer self.randomize = .{};

        self.ticker.tick_real();
        engine.window.tick();
        try self.tick_local_input(engine, app);

        var steps: u32 = 5;
        while (steps > 0 and self.ticker.tick_simulation()) : (steps -= 1) {
            try self.tick_simulation(engine, app);
        }

        self.ticker.tick_animation();
        try self.tick_prepare_render(engine, app);
    }

    fn tick_simulation(self: *@This(), engine: *Engine, app: *App) !void {
        _ = self;
        _ = engine;
        app.telemetry.begin_sample(@src(), "app_state.tick_simulation");
        defer app.telemetry.end_sample();
    }

    fn tick_local_input(self: *@This(), engine: *Engine, app: *App) !void {
        app.telemetry.begin_sample(@src(), "app_state.tick_local_input");
        defer app.telemetry.end_sample();

        const window = engine.window;

        const res = try window.get_res();
        var input = window.input();

        // local input tick
        {
            app.telemetry.begin_sample(@src(), ".local_input");
            defer app.telemetry.end_sample();

            var mouse = &input.mouse;
            var kb = &input.keys;

            const imgui_io = &c.ImGui_GetIO()[0];
            if (imgui_io.WantCaptureMouse) {
                mouse.* = std.mem.zeroes(@TypeOf(input.mouse));
                mouse.x = input.mouse.x;
                mouse.y = input.mouse.y;
                mouse.left = .none;
                mouse.right = .none;
            }
            if (imgui_io.WantCaptureKeyboard) {
                // kb.* = std.mem.zeroes(@TypeOf(kb));
            }

            // TODO: fix
            // if (kb.p.just_pressed()) {
            //     try render_utils.dump_image_to_file(
            //         &app.screen_image,
            //         &engine.graphics,
            //         app.command_pool,
            //         window.extent,
            //         "images",
            //     );
            // }

            // if (mouse.left.just_pressed() and !self.focus) {
            //     self.focus = true;
            //     imgui_io.ConfigFlags |= c.ImGuiConfigFlags_NoMouse;
            //     window.hide_cursor(true);
            // }
            if (kb.escape.just_pressed() and !self.focus) {
                window.queue_close();
            }
            if (kb.escape.just_pressed() and self.focus) {
                self.focus = false;
                imgui_io.ConfigFlags &= ~c.ImGuiConfigFlags_NoMouse;
                window.hide_cursor(false);
            }

            self.mouse.left = mouse.left.pressed();
            self.mouse.x = @intFromFloat(mouse.x);
            self.mouse.y = @intFromFloat(mouse.y);

            self.frame += 1;

            if (!mouse.left.pressed()) {
                mouse.dx = 0;
                mouse.dy = 0;
            }
            self.monitor_rez.width = res.width;
            self.monitor_rez.height = res.height;
        }

        {
            self.camera.eye.x += @as(f32, @floatCast(input.mouse.dx)) / self.params.zoom;
            self.camera.eye.y += @as(f32, @floatCast(input.mouse.dy)) / self.params.zoom;
            self.camera.meta.did_move = @intCast(@intFromBool(@abs(input.mouse.dx) + @abs(input.mouse.dy) > 0.0001));
            self.params.zoom += self.params.zoom * cast(f32, input.mouse.scroll.dy) / 10.0;
        }
    }

    fn tick_prepare_render(self: *@This(), engine: *Engine, app: *App) !void {
        app.telemetry.begin_sample(@src(), "app_state.tick_prepare_render");
        defer app.telemetry.end_sample();

        const window = engine.window;

        // camera tick
        {
            app.telemetry.begin_sample(@src(), ".camera");
            defer app.telemetry.end_sample();

            app.resources.uniform = try ResourceManager.Uniforms.from(self, window);
        }
    }

    pub fn reset_time(self: *@This()) void {
        self.ticker.reset();
        self.frame = 0;
    }
};

pub const GuiState = struct {
    const frame_times_len = 60;
    frame_times: [frame_times_len]f32 = std.mem.zeroes([frame_times_len]f32),
    frame_times_i: usize = 0,
    total: f32 = 0,

    pub fn tick(self: *@This(), app: *App, state: *AppState) !void {
        self.frame_times_i = @rem(self.frame_times_i + 1, self.frame_times.len);
        self.total -= self.frame_times[self.frame_times_i];
        self.frame_times[self.frame_times_i] = state.ticker.real.delta * std.time.ms_per_s;
        self.total += self.frame_times[self.frame_times_i];
        const frametime = self.total / cast(f32, self.frame_times.len);

        c.ImGui_SetNextWindowPos(.{ .x = 5, .y = 5 }, c.ImGuiCond_Once);
        defer c.ImGui_End();
        if (c.ImGui_Begin("SIKE", null, c.ImGuiWindowFlags_None)) {
            c.ImGui_Text("Application average %.3f ms/frame (%.1f FPS)", frametime, std.time.ms_per_s / frametime);

            c.ImGui_Text("State");
            self.editState(app, state);
        }
    }

    fn editState(self: *@This(), app: *App, state: *AppState) void {
        // _ = self;
        // _ = app;

        var reset = false;

        _ = c.ImGui_SliderInt("FPS cap", @ptrCast(&state.fps_cap), 5, 500);
        reset = c.ImGui_SliderInt("spawn count", @ptrCast(&state.spawn_count), 0, 10000) or reset;
        _ = c.ImGui_SliderFloat("zoom", @ptrCast(&state.params.zoom), 0.001, 2.0);
        _ = c.ImGui_SliderInt("particle visual size", @ptrCast(&state.params.particle_visual_size), 1, 100);
        _ = c.ImGui_SliderFloat("particle_z_shrinking_factor", @ptrCast(&state.params.particle_z_shrinking_factor), 0, 1);
        _ = c.ImGui_SliderFloat("particle_z_blur_factor", @ptrCast(&state.params.particle_z_blur_factor), 0, 2);
        _ = c.ImGui_SliderInt("particles type count", @ptrCast(&state.particle_type_count), 1, cast(i32, state.max_particle_type_count));
        _ = c.ImGui_SliderInt("grid size", @ptrCast(&state.params.grid_size), 1, 100);
        _ = c.ImGui_SliderInt("bin size", @ptrCast(&state.bin_size), 4, 200);
        _ = c.ImGui_SliderInt("bin buf size z", @ptrCast(&state.requested_world_size.z), 0, state.bin_size * state.bin_buf_size_z_max);
        reset = c.ImGui_SliderFloat("friction", @ptrCast(&state.friction), 0.0, 5.0) or reset;
        _ = c.ImGui_SliderFloat("entropy", @ptrCast(&state.params.entropy), 0.0, 1.0);
        _ = c.ImGui_SliderFloat("collision_strength_scale", @ptrCast(&state.params.collision_strength_scale), 0, 200);
        _ = c.ImGui_SliderFloat("attraction_strength_scale", @ptrCast(&state.params.attraction_strength_scale), 0, 200);
        _ = c.ImGui_SliderFloat("max_attraction_factor", @ptrCast(&state.params.max_attraction_factor), 1, 200);

        var sim_speed = state.ticker.speed.perc;
        if (c.ImGui_SliderFloat("simulation_speed", @ptrCast(&sim_speed), 0.0, 5.0)) {
            state.ticker.set_speed(sim_speed);
            state.ticker.drop_pending_simtime();
        }

        _ = c.ImGui_SliderInt("step per frame", @ptrCast(&state.steps_per_frame), 1, 20);

        reset = c.ImGui_Button("Reset render state") or reset;

        c.ImGui_Text("scaled time: %.3f", state.ticker.scaled.time_f);
        c.ImGui_Text("physics acctime/step: %.3f", state.ticker.simulation.acctime_f / state.ticker.simulation.step_f);
        c.ImGui_Text("particle count: %d", state.params.particle_count);

        if (c.ImGui_Button("randomize")) {
            state.randomize_particle_forces(app);
            state.randomize_particle_colors(app);
            state.randomize_particle_types(app);
            state.randomize_particle_attrs(app);
        }

        if (c.ImGui_Button("randomize particle forces")) {
            state.randomize_particle_forces(app);
            state.randomize_particle_types(app);
        }

        if (c.ImGui_Button("randomize particle colors")) {
            state.randomize_particle_colors(app);
        }

        if (c.ImGui_Button("randomize particle attrs")) {
            state.randomize_particle_attrs(app);
        }

        {
            c.ImGui_PushID("particle_types");
            defer c.ImGui_PopID();

            c.ImGui_Text("Particles");
            for (app.resources.particle_types[0..state.params.particle_type_count], 0..) |*pt, i| {
                c.ImGui_PushIDInt(@intCast(i));
                defer c.ImGui_PopID();

                c.ImGui_Text("type: %d", i);
                self.editParticleType(pt);
            }
        }
        c.ImGui_Text(" ");
        {
            c.ImGui_PushID("particle_force_matrix");
            defer c.ImGui_PopID();
            c.ImGui_Text("Particle forces");
            for (app.resources.particle_force_matrix[0 .. state.params.particle_type_count * state.params.particle_type_count], 0..) |*pf, i| {
                c.ImGui_PushIDInt(@intCast(i));
                defer c.ImGui_PopID();

                c.ImGui_Text(" ");
                c.ImGui_Text("type: %d vs %d", i / state.params.particle_type_count, i % state.params.particle_type_count);
                self.editParticleForce(pf);
            }
        }

        if (reset) {
            _ = state.cmdbuf_fuse.fuse();
            state.reset_time();
        }
    }

    fn editParticleType(_: *@This(), e: *ResourceManager.ParticleType) void {
        _ = c.ImGui_ColorEdit4("color", e.color.as_buf().ptr, c.ImGuiColorEditFlags_AlphaBar | c.ImGuiColorEditFlags_Float);
        _ = c.ImGui_SliderFloat("particle scale", &e.particle_scale, 0, 1);
    }

    fn editParticleForce(_: *@This(), e: *ResourceManager.ParticleForce) void {
        _ = c.ImGui_SliderFloat("attraction_strength", &e.attraction_strength, -1, 1);
        _ = c.ImGui_SliderFloat("attraction_radius", &e.attraction_radius, 0, 1);
        _ = c.ImGui_SliderFloat("attraction_peak_dist_factor", &e.attraction_peak_dist_factor, 0, 1);
        _ = c.ImGui_SliderFloat("collision_strength", &e.collision_strength, -1, 1);
        _ = c.ImGui_SliderFloat("collision_radius", &e.collision_radius, 0, 1);
    }
};
