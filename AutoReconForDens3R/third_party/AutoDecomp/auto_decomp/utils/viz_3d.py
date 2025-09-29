"""
3D visualization based on plotly.

Adapted from hloc.utils.viz_3d.
"""

from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Union

import numpy as np
import plotly.graph_objects as go
import pycolmap


PlotlySaveMode = Enum("PlotlySaveMode", ["html", "json", "both"])


def to_homogeneous(points):
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def init_figure(
    height: int = 800, width: int = 800, show_axes: bool = False, template="plotly_dark", projection="orthographic"
) -> go.Figure:
    """Initialize a 3D figure."""
    fig = go.Figure()
    if not show_axes:
        axes = dict(
            visible=False,
            showbackground=False,
            showgrid=False,
            showline=False,
            showticklabels=True,
            autorange=True,
        )
    else:
        axes = dict(
            visible=True,
            showbackground=True,
            showgrid=True,
            showline=True,
            showticklabels=True,
            autorange=True,
        )
    fig.update_layout(
        template=template,
        height=height,
        width=width,
        scene_camera=dict(eye=dict(x=0.0, y=-0.1, z=-2), up=dict(x=0, y=-1.0, z=0), projection=dict(type=projection)),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.1),
    )
    return fig


def plot_points(
    fig: go.Figure,
    pts: np.ndarray,
    color: Union[str, Iterable] = "rgba(255, 0, 0, 1)",
    ps: int = 2,
    lw: int = 0,
    marker_symbol: Optional[Union[List[int], str]] = None,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
    show_legend=True,
    **kwargs,
):
    """Plot a set of 3D points."""
    if not isinstance(color, str):
        assert len(color) == len(pts)

    x, y, z = pts.T
    tr = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name=name,
        legendgroup=name,
        showlegend=show_legend,
        **kwargs,
        marker=dict(
            size=ps, color=color, colorscale=colorscale, symbol=marker_symbol, line=dict(width=lw, color=color)
        ),
    )
    fig.add_trace(tr)


def plot_mesh(
    fig: go.Figure,
    vertices: np.array,  # (N, 3)
    faces: np.array,  # (N, 3)
    color: str = "lightpink",
    opacity: float = 0.4,
    name: Optional[str] = None,
    show_legend: bool = False,
):
    vertices, faces = vertices.T, faces.T
    tr = go.Mesh3d(
        x=vertices[0],
        y=vertices[1],
        z=vertices[2],
        i=faces[0],
        j=faces[1],
        k=faces[2],
        color=color,
        opacity=opacity,
        name=name,
        legendgroup=name if show_legend else None,
        showlegend=show_legend,
    )
    fig.add_trace(tr)


def plot_cube(
    fig: go.Figure,
    vertices: np.array = None,  # (8, 3)
    center: np.array = None,  # (3,)
    extent: np.array = None,  # (3,) NOTE: not half_extent!
    rotation: np.array = None,  # (3, 3), represent oriented bbox (rotation: aabb => oriented)
    color: str = "lightpink",
    opacity: float = 0.4,
    name: Optional[str] = None,
    show_legend: bool = False,
    hoverinfo: str = "none",
    hovertext: Optional[str] = None,
):
    if vertices is None:
        assert center is not None and extent is not None
        hx, hy, hz = (extent / 2).tolist()
        x, y, z = center.tolist()
        vertices = np.array(
            [
                [x - hx, y + hy, z + hz],
                [x + hx, y + hy, z + hz],
                [x + hx, y + hy, z - hz],
                [x - hx, y + hy, z - hz],
                [x - hx, y - hy, z + hz],
                [x + hx, y - hy, z + hz],
                [x + hx, y - hy, z - hz],
                [x - hx, y - hy, z - hz],
            ]
        )
        if rotation is not None:
            vertices = vertices @ rotation.T

    vertices = vertices.T

    if hoverinfo == "none":
        pass
    elif hoverinfo == "text":
        if hovertext is None:
            hovertext = name
    else:
        raise ValueError(f"Unknown hoverinfo: {hoverinfo}")

    tr = go.Mesh3d(
        x=vertices[0],
        y=vertices[1],
        z=vertices[2],
        i=[0, 0, 0, 0, 1, 1, 0, 0, 3, 3, 4, 4],
        j=[1, 3, 1, 4, 2, 5, 3, 4, 2, 7, 7, 5],
        k=[2, 2, 5, 5, 6, 6, 7, 7, 6, 6, 6, 6],
        color=color,
        opacity=opacity,
        flatshading=True,
        name=name,
        legendgroup=name if show_legend else None,
        showlegend=show_legend,
        hoverinfo=hoverinfo,
        hovertext=hovertext,
    )
    fig.add_trace(tr)


def plot_sphere(
    fig: go.Figure,
    center: np.ndarray = np.zeros((3,)),  # (3, )
    radius: float = 1,
    color: str = "lightpink",
    opacity: float = 0.4,
    name: Optional[str] = None,
    show_legend: bool = False,
):
    phi = np.linspace(0, 2 * np.pi)
    theta = np.linspace(-np.pi / 2, np.pi / 2)
    phi, theta = np.meshgrid(phi, theta)
    x = radius * np.cos(theta) * np.sin(phi) + center[0]
    y = radius * np.sin(theta) * np.sin(phi) + center[1]
    z = radius * np.cos(phi) + center[2]

    tr = go.Mesh3d(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        flatshading=False,
        alphahull=0,  # -1: delaunay triangulation | >0: alpha-shape | 0: convex hull
        name=name,
        color=color,
        opacity=opacity,
        legendgroup=name if show_legend else None,
        showlegend=show_legend,
    )
    fig.add_trace(tr)


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,  # c2w_33
    t: np.ndarray,  # c2w_3
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    size: float = 1.0,
    T: Optional[np.ndarray] = None,  # T_w0=>w1_44
):
    """Plot a camera frustum from pose and intrinsic matrix."""
    if T is not None:
        R = T[:3, :3] @ R
        t = T[:3, :3] @ t + T[:3, -1]

    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t

    x, y, z = corners.T
    rect = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        line=dict(color=color),
        legendgroup=legendgroup,
        name=name,
        marker=dict(size=0.0001),
        showlegend=False,
    )
    fig.add_trace(rect)

    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    pyramid = go.Mesh3d(x=x, y=y, z=z, color=color, i=i, j=j, k=k, legendgroup=legendgroup, name=name, showlegend=False)
    fig.add_trace(pyramid)
    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[i] for i in triangles.reshape(-1)])

    x, y, z = tri_points.T
    pyramid = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        legendgroup=legendgroup,
        name=name,
        line=dict(color=color, width=1),
        showlegend=False,
    )
    fig.add_trace(pyramid)


def plot_camera_colmap(
    fig: go.Figure, image: pycolmap.Image, camera: pycolmap.Camera, name: Optional[str] = None, **kwargs
):
    """Plot a camera frustum from PyCOLMAP objects"""
    plot_camera(
        fig,
        image.rotmat().T,
        image.projection_center(),
        camera.calibration_matrix(),
        name=name or str(image.image_id),
        **kwargs,
    )


def plot_cameras(fig: go.Figure, reconstruction: pycolmap.Reconstruction, **kwargs):
    """Plot a camera as a cone with camera frustum."""
    for image_id, image in reconstruction.images.items():
        plot_camera_colmap(fig, image, reconstruction.cameras[image.camera_id], **kwargs)


def plot_coordinate_frames(
    fig: go.Figure,
    center: np.ndarray = np.zeros((3,)),
    length: Union[int, np.ndarray] = 1,
    line_width: int = 8,
    opacity: float = 0.5,
    # name: Optional[str] = None,
    # legendgroup: Optional[str] = None,
):
    # TODO: line opacity
    dirs = np.eye(3) * length
    trace_o0 = go.Scatter3d(
        x=[center[0], dirs[0, 0]],
        y=[center[1], dirs[0, 1]],
        z=[center[2], dirs[0, 2]],
        hoverinfo="skip",
        mode="lines",
        line=dict(color="red", width=line_width),
        name="x",
    )
    trace_o1 = go.Scatter3d(
        x=[center[0], dirs[1, 0]],
        y=[center[1], dirs[1, 1]],
        z=[center[2], dirs[1, 2]],
        hoverinfo="skip",
        mode="lines",
        line=dict(color="green", width=line_width),
        name="y",
    )
    trace_o2 = go.Scatter3d(
        x=[center[0], dirs[2, 0]],
        y=[center[1], dirs[2, 1]],
        z=[center[2], dirs[2, 2]],
        hoverinfo="skip",
        mode="lines",
        line=dict(color="blue", width=line_width),
        name="z",
    )
    traces = [trace_o0, trace_o1, trace_o2]
    # TODO: merge traces into one
    fig.add_traces(traces)


def plot_reconstruction(
    fig: go.Figure,
    rec: pycolmap.Reconstruction,
    max_reproj_error: float = 6.0,
    min_track_length: int = 2,
    points: bool = True,
    points_color: Optional[str] = "rgb(0, 0, 255)",
    points_name: Optional[str] = None,
    points_size: int = 4,
    cameras_color: str = "rgb(255, 0, 0)",
    cameras: bool = True,
    cameras_name: Optional[str] = "cameras",
    camera_scale: float = 1.0,
):
    # Filter outliers
    bbs = rec.compute_bounding_box(0.001, 0.999)

    # Filter points, use original reproj error here
    kept_p3Ds = {
        idx: p3D.xyz
        for idx, p3D in rec.points3D.items()
        if (
            (p3D.xyz >= bbs[0]).all()
            and (p3D.xyz <= bbs[1]).all()
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    }

    if points:
        xyzs = np.array(list(kept_p3Ds.values()))
        if points_color is None:
            points_color = np.array([rec.points3D[idx].color for idx in kept_p3Ds.keys()])
        plot_points(fig, np.array(xyzs), color=points_color, ps=points_size, name=points_name)
    if cameras:
        plot_cameras(fig, rec, color=cameras_color, name=cameras_name, size=camera_scale)


def save_fig(fig, save_dir: Path, fig_name: str, mode: Union[str, PlotlySaveMode] = PlotlySaveMode.html):
    if isinstance(mode, str):
        mode = PlotlySaveMode[mode]
    
    if mode == PlotlySaveMode.html or mode == PlotlySaveMode.both:
        fig.write_html(save_dir / f"{fig_name}.html", full_html=False, include_plotlyjs='cdn')
    if mode == PlotlySaveMode.json or mode == PlotlySaveMode.both:
        fig.write_json(save_dir / f"{fig_name}.json", pretty=True)
