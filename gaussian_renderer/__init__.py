import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from time import time as get_time

def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, stage="fine", cam_type=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!

    Returns a dict containing:
      - render:         the rendered image
      - viewspace_points: 2D screen-space points (for gradient tracking)
      - visibility_filter: mask of visible Gaussians
      - radii:             screen-space radii
      - depth:             depth map
      - deformation_data:  dict with original params and deltas
    """
    # 1. Create zero tensor to capture 2D screen-space means gradients
    screenspace_points = torch.zeros_like(
        pc.get_xyz, dtype=pc.get_xyz.dtype,
        requires_grad=True, device="cuda"
    )
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 2. Base 3D means
    means3D = pc.get_xyz

    # 3. Setup rasterization settings & time tag
    if cam_type != "PanopticSports":
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.cuda(),
            projmatrix=viewpoint_camera.full_proj_transform.cuda(),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center.cuda(),
            prefiltered=False,
            debug=pipe.debug
        )
        time_tag = torch.tensor(viewpoint_camera.time) \
                          .to(means3D.device) \
                          .repeat(means3D.shape[0], 1)
    else:
        raster_settings = viewpoint_camera['camera']
        time_tag = torch.tensor(viewpoint_camera['time']) \
                          .to(means3D.device) \
                          .repeat(means3D.shape[0], 1)

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 4. Prepare other parameters
    opacity = pc._opacity
    shs     = pc.get_features
    scales     = None
    rotations  = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales    = pc._scaling
        rotations = pc._rotation

    # 5. Handle deformation stage
    deformation_data = None
    if "coarse" in stage:
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
            means3D, scales, rotations, opacity, shs

    elif "fine" in stage:
        # --- Clone originals before deformation ---
        orig_means3D   = means3D.clone()
        orig_scales    = scales.clone()    if scales    is not None else None
        orig_rotations = rotations.clone() if rotations is not None else None
        orig_opacity   = opacity.clone()
        orig_shs       = shs.clone()
        orig_time      = time_tag.clone()

        # --- Apply deformation ---
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = \
            pc._deformation(
                means3D, scales, rotations,
                opacity, shs, time_tag
            )

        # --- Compute deltas ---
        delta_means3D   = means3D_final - orig_means3D
        delta_scales    = (scales_final - orig_scales)       if orig_scales    is not None else None
        delta_rotations = (rotations_final - orig_rotations) if orig_rotations is not None else None
        delta_opacity   = opacity_final - orig_opacity
        delta_shs       = shs_final     - orig_shs

        # --- Pack into deformation_data dict ---
        deformation_data = {
            'orig_means3D':   orig_means3D,
            'orig_scales':    orig_scales,
            'orig_rotations': orig_rotations,
            'orig_opacity':   orig_opacity,
            'orig_shs':       orig_shs,
            'orig_time':      orig_time,
            'delta_means3D':   delta_means3D,
            'delta_scales':    delta_scales,
            'delta_rotations': delta_rotations,
            'delta_opacity':   delta_opacity,
            'delta_shs':       delta_shs
        }

    else:
        raise NotImplementedError(f"Unknown stage: {stage}")

    # 6. Activate transformed parameters
    scales_final    = pc.scaling_activation(scales_final)    if scales_final    is not None else None
    rotations_final = pc.rotation_activation(rotations_final) if rotations_final is not None else None
    opacity         = pc.opacity_activation(opacity_final)

    # 7. Prepare colors
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2) \
                        .view(-1, 3, (pc.max_sh_degree+1)**2)
            dirs = (pc.get_xyz - viewpoint_camera.camera_center.cuda()
                    .repeat(pc.get_features.shape[0], 1))
            dirs_norm = dirs / dirs.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dirs_norm)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors_precomp = override_color

    # 8. Rasterize to image, radii, depth
    rendered_image, radii, depth = rasterizer(
        means3D=means3D_final,
        means2D=screenspace_points,
        shs=shs_final,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales_final,
        rotations=rotations_final,
        cov3D_precomp=cov3D_precomp
    )

    # 9. Return full result
    return {
        "render":           rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii":             radii,
        "depth":             depth,
        "deformation_data":  deformation_data
    }
