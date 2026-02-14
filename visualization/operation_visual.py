"""
Operation visuals with continuous data flow.

Every visual supports:
  - appear_t (0→1): Input matrix A flies in from `from_origin_a` to `origin_a`
  - t (0→1): The actual operation animation
  - depart_t (0→1): Result C flies from `origin_c` to `to_origin_c`

MatmulVisual follows the descent+dissolve+collapse pattern:
  1. B column descends from above A to A's column positions
  2. B splits into per-element copies that dissolve INTO A (weights consumed)
  3. Products emerge from A rows and collapse to C positions (summation)
"""
import numpy as np
from visualization.colormap import matrix_to_colors
from animation.easing import ease_in_out_cubic, ease_out_cubic, smoothstep


def _box_pos(origin, row, col, spacing):
    return np.array([
        origin[0] + col * spacing,
        origin[1] - row * spacing,
        origin[2]
    ], dtype=np.float32)


def _make_instance(pos, color, scale, scale_z=None):
    sz = scale_z if scale_z is not None else scale
    return np.array([
        pos[0], pos[1], pos[2],
        color[0], color[1], color[2], color[3],
        scale, scale, sz
    ], dtype=np.float32)


def _lerp_pos(a, b, t):
    return a + (b - a) * t


def _wave_t(base_t, row, col, rows, cols, stagger=0.3):
    """Per-element staggered timing with diagonal wave (top-left → bottom-right).
    Returns eased t in [0, 1] for this element."""
    diag_max = max(rows + cols - 2, 1)
    delay = ((row + col) / diag_max) * stagger
    raw = np.clip((base_t - delay) / max(1.0 - stagger, 0.1), 0.0, 1.0)
    return ease_in_out_cubic(raw)


class MatmulVisual:
    """C = A × B with descent + dissolve + collapse animation.

    appear: A flies in from from_origin_a (or from_origin_a_slices for multi-source)
    compute: per column j of B (staggered):
      1. B[:,j] descends from above to center of A's columns
      2. B splits into m copies that move to each A[i,p] and dissolve (absorbed)
      3. Products emerge from A rows and fly to C[i,j]
    depart: C flies to to_origin_c
    """

    def __init__(self, A, B, C, origin_a, origin_b, origin_c,
                 box_size=0.4, gap=0.1, box_size_b=None, gap_b=None,
                 box_size_c=None, gap_c=None):
        self.A = A
        self.B = B
        self.C = C
        self.origin_a = np.array(origin_a, dtype=np.float32)
        self.origin_b = np.array(origin_b, dtype=np.float32)
        self.origin_c = np.array(origin_c, dtype=np.float32)
        self.bs = box_size
        self.sp = box_size + gap
        self.bs_b = box_size_b if box_size_b is not None else box_size
        self.sp_b = (box_size_b or box_size) + (gap_b if gap_b is not None else gap)
        self.bs_c = box_size_c if box_size_c is not None else box_size
        self.sp_c = (box_size_c or box_size) + (gap_c if gap_c is not None else gap)

        # Flow connections (set by scene)
        self.from_origin_a = None  # Single origin for A fly-in
        self.from_origin_a_slices = None  # List of (col_start, col_end, origin) for multi-source
        self.from_origin_b = None  # Single origin for B fly-in
        self.seamless_b = True  # False = fade-in instead of full alpha (e.g. shape change)
        self.transpose_fly_b = False  # True = B elements fly from transposed positions (e.g. K_h → K_h^T)
        self.to_origin_c = None

        # Animation state (set by Stage)
        self.appear_t = 0.0
        self.t = 0.0
        self.depart_t = 0.0
        self.alpha = 1.0
        self.phase_group = 0
        self.is_stage_output = False
        self.output_alpha_mult = 1.0  # 0 = hide C instantly during fade

        # Optional: override color normalization range (for sliced matrices)
        self.color_vmax_a = None
        self.color_vmax_b = None
        self.color_vmax_c = None

        self._colors_a = None
        self._colors_b = None
        self._colors_c = None

    def _ensure_colors(self):
        if self._colors_a is None:
            self._colors_a = matrix_to_colors(self.A, 1.0, vmax=self.color_vmax_a)
            self._colors_b = matrix_to_colors(self.B, 1.0, vmax=self.color_vmax_b)
            self._colors_c = matrix_to_colors(self.C, 1.0, vmax=self.color_vmax_c)

    def get_instance_data(self):
        if self.alpha < 0.01:
            return np.zeros((0, 10), dtype=np.float32)

        self._ensure_colors()
        m, k = self.A.shape
        _, n = self.C.shape
        instances = []

        # === Phase: A flying in ===
        if self.appear_t < 1.0 and self.t <= 0.0:
            if self.from_origin_a_slices is not None:
                # Multi-source fly-in (e.g., head outputs merging into concat)
                # Seamless crossfade: alpha = appear_t complements prev stage fade
                for col_start, col_end, slice_origin in self.from_origin_a_slices:
                    slice_w = col_end - col_start
                    for i in range(m):
                        for local_p in range(slice_w):
                            p = col_start + local_p
                            at = _wave_t(self.appear_t, i, local_p, m, slice_w)
                            src = _box_pos(slice_origin, i, local_p, self.sp)
                            dst = _box_pos(self.origin_a, i, p, self.sp)
                            pos = _lerp_pos(src, dst, at)
                            color = self._colors_a[i, p].copy()
                            color[3] = self.alpha
                            instances.append(_make_instance(pos, color, self.bs))
            else:
                # Single-source fly-in
                from_o = self.from_origin_a if self.from_origin_a is not None else self.origin_a
                seamless = self.from_origin_a is not None
                for i in range(m):
                    for p in range(k):
                        at = _wave_t(self.appear_t, i, p, m, k)
                        src = _box_pos(from_o, i, p, self.sp)
                        dst = _box_pos(self.origin_a, i, p, self.sp)
                        pos = _lerp_pos(src, dst, at)
                        color = self._colors_a[i, p].copy()
                        if seamless:
                            color[3] = self.alpha
                            instances.append(_make_instance(pos, color, self.bs))
                        else:
                            color[3] = self.alpha * at
                            instances.append(_make_instance(pos, color, self.bs * (0.3 + 0.7 * at)))

            # B flies in from from_origin_b (or fades in at origin_b)
            # seamless_b=True + from_origin_b: full alpha crossfade
            # seamless_b=False + from_origin_b: fly from origin but fade in
            # no from_origin_b: delayed fade-in at origin_b
            b_has_origin = self.from_origin_b is not None
            b_seamless = b_has_origin and self.seamless_b
            if b_has_origin or self.appear_t > 0.3:
                from_b = self.from_origin_b if b_has_origin else self.origin_b
                b_norm = self.appear_t if b_has_origin else smoothstep(0.3, 1.0, self.appear_t)
                for r in range(k):
                    for c in range(n):
                        bt = _wave_t(b_norm, r, c, k, n, 0.2)
                        if self.transpose_fly_b and b_has_origin:
                            # Transpose animation: B[r,c] = K_h^T[r,c] = K_h[c,r]
                            # Source at pre-transpose position (row=c, col=r)
                            src = _box_pos(from_b, c, r, self.sp_b)
                        else:
                            src = _box_pos(from_b, r, c, self.sp_b)
                        dst = _box_pos(self.origin_b, r, c, self.sp_b)
                        pos = _lerp_pos(src, dst, bt)
                        color = self._colors_b[r, c].copy()
                        if b_seamless:
                            # Smooth alpha transition: 1.0 → 0.9 as element arrives
                            color[3] = self.alpha * (1.0 - bt * 0.1)
                            instances.append(_make_instance(pos, color, self.bs_b))
                        else:
                            color[3] = self.alpha * bt * 0.9
                            instances.append(_make_instance(pos, color, self.bs_b * (0.3 + 0.7 * bt)))

            return np.array(instances, dtype=np.float32) if instances else np.zeros((0, 10), dtype=np.float32)

        # === Phase: Compute (fly from B → dissolve at A → collapse to C) ===
        if self.t > 0.0 or self.appear_t >= 1.0:
            t = max(0.0, self.t)

            col_window = min(3.0 / max(n, 1), 0.5)
            col_stride = (1.0 - col_window) / max(n - 1, 1) if n > 1 else 0.0

            # --- 1. Render A matrix (always visible during compute) ---
            for i in range(m):
                for p in range(k):
                    pos = _box_pos(self.origin_a, i, p, self.sp)
                    color = self._colors_a[i, p].copy()
                    color[3] = self.alpha
                    instances.append(_make_instance(pos, color, self.bs))

            # --- 2. B at origin_b: columns depart as they're used ---
            for r_b in range(k):
                for c_b in range(n):
                    cs = c_b * col_stride
                    ce = cs + col_window
                    if t >= ce:
                        # Column fully used → gone
                        continue
                    elif t >= cs:
                        # Column currently departing → fade out as B flies away
                        depart_frac = (t - cs) / col_window
                        if depart_frac < 0.25:
                            leave_t = depart_frac / 0.25
                            fade = 1.0 - leave_t
                            pos = _box_pos(self.origin_b, r_b, c_b, self.sp_b)
                            color = self._colors_b[r_b, c_b].copy()
                            color[3] = self.alpha * fade * 0.8
                            scale = self.bs_b * (1.0 - leave_t * 0.6)
                            instances.append(_make_instance(pos, color, scale))
                    else:
                        # Column not yet used → full opacity
                        pos = _box_pos(self.origin_b, r_b, c_b, self.sp_b)
                        color = self._colors_b[r_b, c_b].copy()
                        color[3] = self.alpha * 0.9
                        instances.append(_make_instance(pos, color, self.bs_b))

            # --- 3. Column sweep: B flies to A elements → dissolve → collapse to C ---
            for j in range(n):
                col_start = j * col_stride
                col_end = col_start + col_window

                if t < col_start:
                    continue

                if t > col_end:
                    # Column done: show solid C element
                    if self.output_alpha_mult > 0.01:
                        for i in range(m):
                            c_pos = _box_pos(self.origin_c, i, j, self.sp_c)
                            color = self._colors_c[i, j].copy()
                            color[3] = self.alpha * self.output_alpha_mult
                            instances.append(_make_instance(c_pos, color, self.bs_c))
                    continue

                sub_t = (t - col_start) / col_window

                if sub_t < 0.20:
                    # Phase 1: B[:,j] (ONE column, k boxes) flies from origin_b
                    #          to A's first row (column → row rotation)
                    for p in range(k):
                        fly_t = _wave_t(sub_t / 0.20, p, 0, k, 1, 0.3)
                        b_src = _box_pos(self.origin_b, p, j, self.sp_b)
                        a_row0 = _box_pos(self.origin_a, 0, p, self.sp)

                        pos = _lerp_pos(b_src, a_row0, fly_t)
                        color = self._colors_b[p, j].copy()
                        color[3] = self.alpha * (0.5 + 0.5 * fly_t)
                        xy_s = self.bs_b + (self.bs * 0.75 - self.bs_b) * fly_t
                        instances.append(_make_instance(pos, color, max(0.05, xy_s), scale_z=self.bs * 1.3))

                elif sub_t < 0.55:
                    # Phase 2: B elements step down through A's rows, dwelling at each
                    # row to emphasize per-row dot product computation.
                    phase2_t = (sub_t - 0.20) / 0.35  # 0→1
                    num_steps = max(m - 1, 1)
                    stagger_total = min(0.3, 0.025 * (k - 1))
                    anim_portion = max(1.0 - stagger_total, 0.3)
                    dwell_ratio = 0.45  # fraction of each step spent pausing at a row

                    for p in range(k):
                        p_delay = (p / max(k - 1, 1)) * stagger_total if k > 1 else 0.0
                        p_t = np.clip((phase2_t - p_delay) / anim_portion, 0.0, 1.0)

                        step_f = p_t * num_steps
                        step_idx = min(int(step_f), num_steps - 1)
                        step_frac = min(step_f - step_idx, 1.0)

                        from_row = step_idx
                        to_row = min(step_idx + 1, m - 1)

                        if step_frac < dwell_ratio:
                            # Dwell: element pauses at current row
                            eased = 0.0
                            dwell_t = step_frac / dwell_ratio  # 0→1 within dwell
                            pulse = 1.0 + 0.15 * np.sin(dwell_t * np.pi)
                            elem_scale = self.bs * 0.75 * pulse
                            elem_alpha = self.alpha * 0.95
                        else:
                            # Move: element transitions to next row
                            move_t = (step_frac - dwell_ratio) / (1.0 - dwell_ratio)
                            eased = ease_in_out_cubic(move_t)
                            elem_scale = self.bs * 0.75
                            elem_alpha = self.alpha * 0.9

                        src_pos = _box_pos(self.origin_a, from_row, p, self.sp)
                        dst_pos = _box_pos(self.origin_a, to_row, p, self.sp)
                        active_pos = _lerp_pos(src_pos, dst_pos, eased)
                        color = self._colors_b[p, j].copy()
                        color[3] = elem_alpha
                        instances.append(_make_instance(active_pos, color, elem_scale, scale_z=self.bs * 1.3))

                        # Row highlight glow during dwell
                        if step_frac < dwell_ratio:
                            a_pos = _box_pos(self.origin_a, from_row, p, self.sp)
                            glow_a = self.alpha * 0.2 * np.sin(dwell_t * np.pi)
                            glow_color = np.array([1.0, 1.0, 1.0, glow_a], dtype=np.float32)
                            instances.append(_make_instance(a_pos, glow_color, self.bs * 1.05))

                        # Copies stamped at previously visited rows
                        for si in range(from_row):
                            s_pos = _box_pos(self.origin_a, si, p, self.sp)
                            s_color = self._colors_b[p, j].copy()
                            s_color[3] = self.alpha * 0.45
                            instances.append(_make_instance(s_pos, s_color, self.bs * 0.6, scale_z=self.bs * 1.3))

                elif sub_t < 0.65:
                    # Phase 3: All B copies at every A[i,p] dissolve (multiplication)
                    dissolve_t = ease_in_out_cubic((sub_t - 0.55) / 0.10)

                    for p in range(k):
                        for i in range(m):
                            a_pos = _box_pos(self.origin_a, i, p, self.sp)
                            color = self._colors_b[p, j].copy()
                            fade = max(0.0, 1.0 - dissolve_t)
                            color[3] = self.alpha * fade * 0.7
                            xy_s = max(0.01, self.bs * 0.75 * (1.0 - dissolve_t * 0.8))
                            instances.append(_make_instance(a_pos, color, xy_s, scale_z=self.bs * 1.3))

                else:
                    # Phase 4: Products emerge from A rows and fly to C[i,j]
                    if self.output_alpha_mult > 0.01:
                        for i in range(m):
                            collapse_t = _wave_t((sub_t - 0.65) / 0.35, i, 0, m, 1, 0.3)
                            row_center = np.array([
                                self.origin_a[0] + (k - 1) * self.sp / 2,
                                self.origin_a[1] - i * self.sp,
                                self.origin_a[2]
                            ], dtype=np.float32)
                            c_target = _box_pos(self.origin_c, i, j, self.sp_c)
                            pos = _lerp_pos(row_center, c_target, collapse_t)
                            color = self._colors_c[i, j].copy()
                            color[3] = self.alpha * self.output_alpha_mult * (0.3 + 0.7 * collapse_t)
                            scale = self.bs_c * (0.3 + 0.7 * collapse_t)
                            instances.append(_make_instance(pos, color, scale))

        # === Phase: C departing ===
        if self.depart_t > 0.0 and self.to_origin_c is not None and self.output_alpha_mult > 0.01:
            dt_val = ease_in_out_cubic(self.depart_t)
            for i in range(m):
                for j in range(n):
                    src = _box_pos(self.origin_c, i, j, self.sp_c)
                    dst = _box_pos(self.to_origin_c, i, j, self.sp_c)
                    pos = _lerp_pos(src, dst, dt_val)
                    color = self._colors_c[i, j].copy()
                    color[3] = self.alpha * self.output_alpha_mult
                    instances.append(_make_instance(pos, color, self.bs_c))

        if not instances:
            return np.zeros((0, 10), dtype=np.float32)
        return np.array(instances, dtype=np.float32)


class AddVisual:
    """C = A + B with flow:
    appear: A flies in from from_origin_a, B flies in from from_origin_b
    compute: B overlays on A → merge → result appears at C
    depart: C flies to to_origin_c
    """

    def __init__(self, A, B, C, origin_a, origin_b, origin_c,
                 box_size=0.4, gap=0.1):
        self.A = A
        self.B = B
        self.C = C
        self.origin_a = np.array(origin_a, dtype=np.float32)
        self.origin_b = np.array(origin_b, dtype=np.float32)
        self.origin_c = np.array(origin_c, dtype=np.float32)
        self.bs = box_size
        self.sp = box_size + gap

        self.from_origin_a = None
        self.from_origin_b = None
        self.to_origin_c = None
        self.seamless_a = True   # False for skip connections (fade in instead)
        self.seamless_b = True

        self.appear_t = 0.0
        self.t = 0.0
        self.depart_t = 0.0
        self.alpha = 1.0
        self.phase_group = 0
        self.is_stage_output = False
        self.output_alpha_mult = 1.0

        self._colors_a = None
        self._colors_b = None
        self._colors_c = None

    def _ensure_colors(self):
        if self._colors_a is None:
            self._colors_a = matrix_to_colors(self.A, 1.0)
            self._colors_b = matrix_to_colors(self.B, 1.0)
            self._colors_c = matrix_to_colors(self.C, 1.0)

    def get_instance_data(self):
        if self.alpha < 0.01:
            return np.zeros((0, 10), dtype=np.float32)

        self._ensure_colors()
        rows, cols = self.A.shape
        instances = []

        # === Appear: A and B fly in ===
        if self.appear_t < 1.0 and self.t <= 0.0:
            from_a = self.from_origin_a if self.from_origin_a is not None else self.origin_a
            from_b = self.from_origin_b if self.from_origin_b is not None else self.origin_b
            seamless_a = self.seamless_a and self.from_origin_a is not None
            seamless_b = self.seamless_b and self.from_origin_b is not None
            for r in range(rows):
                for c in range(cols):
                    at = _wave_t(self.appear_t, r, c, rows, cols)
                    src_a = _box_pos(from_a, r, c, self.sp)
                    dst_a = _box_pos(self.origin_a, r, c, self.sp)
                    pos_a = _lerp_pos(src_a, dst_a, at)
                    col_a = self._colors_a[r, c].copy()
                    if seamless_a:
                        col_a[3] = self.alpha
                        instances.append(_make_instance(pos_a, col_a, self.bs))
                    else:
                        col_a[3] = self.alpha * at
                        instances.append(_make_instance(pos_a, col_a, self.bs * (0.3 + 0.7 * at)))

                    bt = _wave_t(max(0, (self.appear_t - 0.15) / 0.85), r, c, rows, cols)
                    src_b = _box_pos(from_b, r, c, self.sp)
                    dst_b = _box_pos(self.origin_b, r, c, self.sp)
                    pos_b = _lerp_pos(src_b, dst_b, bt)
                    col_b = self._colors_b[r, c].copy()
                    if seamless_b:
                        # Smooth alpha transition: 1.0 → 0.8 as element arrives
                        col_b[3] = self.alpha * (1.0 - bt * 0.2)
                        instances.append(_make_instance(pos_b, col_b, self.bs))
                    else:
                        col_b[3] = self.alpha * bt * 0.8
                        instances.append(_make_instance(pos_b, col_b, self.bs * (0.3 + 0.7 * bt)))
            return np.array(instances, dtype=np.float32) if instances else np.zeros((0, 10), dtype=np.float32)

        # === Compute: B descends onto A, merge, result to C ===
        if self.t > 0.0 or self.appear_t >= 1.0:
            ct = max(0.0, self.t)
            P1 = 0.30; P2 = 0.60

            for r in range(rows):
                for c in range(cols):
                    # Per-element stagger: starts later, but all finish at ct=1.0
                    stagger = 0.15
                    elem_delay = ((r + c) / max(rows + cols - 2, 1)) * stagger
                    ect = np.clip((ct - elem_delay) / max(1.0 - elem_delay, 0.1), 0.0, 1.0)

                    a_pos = _box_pos(self.origin_a, r, c, self.sp)
                    b_pos_src = _box_pos(self.origin_b, r, c, self.sp)
                    c_pos = _box_pos(self.origin_c, r, c, self.sp)

                    if ect < P1:
                        fly = ease_out_cubic(ect / P1)
                        instances.append(_make_instance(a_pos, self._set_alpha(self._colors_a[r, c], self.alpha), self.bs))
                        b_target = np.array([a_pos[0], a_pos[1] + 0.4, a_pos[2] + 0.3], dtype=np.float32)
                        b_current = _lerp_pos(b_pos_src, b_target, fly)
                        instances.append(_make_instance(b_current, self._set_alpha(self._colors_b[r, c], self.alpha * 0.8), self.bs))
                    elif ect < P2:
                        merge = (ect - P1) / (P2 - P1)
                        instances.append(_make_instance(a_pos, self._set_alpha(self._colors_a[r, c], self.alpha * (1.0 - merge * 0.6)), self.bs * (1.0 - merge * 0.4)))
                        b_at_a = np.array([a_pos[0], a_pos[1] + 0.4 * (1.0 - merge), a_pos[2] + 0.3 * (1.0 - merge)], dtype=np.float32)
                        instances.append(_make_instance(b_at_a, self._set_alpha(self._colors_b[r, c], self.alpha * 0.8 * (1.0 - merge * 0.8)), self.bs * (1.0 - merge * 0.6)))
                        if merge > 0.3 and self.output_alpha_mult > 0.01:
                            c_alpha = smoothstep(0.3, 1.0, merge)
                            instances.append(_make_instance(a_pos, self._set_alpha(self._colors_c[r, c], self.alpha * c_alpha * self.output_alpha_mult), self.bs * (0.5 + 0.5 * c_alpha)))
                    else:
                        if self.output_alpha_mult > 0.01:
                            move = ease_in_out_cubic((ect - P2) / (1.0 - P2))
                            pos = _lerp_pos(a_pos, c_pos, move)
                            instances.append(_make_instance(pos, self._set_alpha(self._colors_c[r, c], self.alpha * self.output_alpha_mult), self.bs))

        # === Depart ===
        if self.depart_t > 0.0 and self.to_origin_c is not None and self.output_alpha_mult > 0.01:
            dt_val = ease_in_out_cubic(self.depart_t)
            for r in range(rows):
                for c in range(cols):
                    src = _box_pos(self.origin_c, r, c, self.sp)
                    dst = _box_pos(self.to_origin_c, r, c, self.sp)
                    pos = _lerp_pos(src, dst, dt_val)
                    instances.append(_make_instance(pos, self._set_alpha(self._colors_c[r, c], self.alpha * self.output_alpha_mult), self.bs))

        if not instances:
            return np.zeros((0, 10), dtype=np.float32)
        return np.array(instances, dtype=np.float32)

    def _set_alpha(self, color, alpha):
        c = color.copy()
        c[3] = alpha
        return c


class ActivationVisual:
    """In-place transform with flow.
    appear: values fly in from from_origin
    compute: row-by-row color transition
    depart: result flies to to_origin
    """

    def __init__(self, pre, post, origin, box_size=0.4, gap=0.1):
        self.pre = pre
        self.post = post
        self.origin = np.array(origin, dtype=np.float32)
        self.bs = box_size
        self.sp = box_size + gap

        self.from_origin = None
        self.to_origin = None

        self.appear_t = 0.0
        self.t = 0.0
        self.depart_t = 0.0
        self.alpha = 1.0
        self.phase_group = 0
        self.is_stage_output = False
        self.output_alpha_mult = 1.0

        self._colors_pre = None
        self._colors_post = None

    def _ensure_colors(self):
        if self._colors_pre is None:
            self._colors_pre = matrix_to_colors(self.pre, 1.0)
            self._colors_post = matrix_to_colors(self.post, 1.0)

    def get_instance_data(self):
        if self.alpha < 0.01 or self.output_alpha_mult < 0.01:
            return np.zeros((0, 10), dtype=np.float32)

        self._ensure_colors()
        rows, cols = self.pre.shape
        instances = []

        # === Appear ===
        if self.appear_t < 1.0 and self.t <= 0.0:
            from_o = self.from_origin if self.from_origin is not None else self.origin
            seamless = self.from_origin is not None
            for r in range(rows):
                for c in range(cols):
                    at = _wave_t(self.appear_t, r, c, rows, cols)
                    src = _box_pos(from_o, r, c, self.sp)
                    dst = _box_pos(self.origin, r, c, self.sp)
                    pos = _lerp_pos(src, dst, at)
                    color = self._colors_pre[r, c].copy()
                    if seamless:
                        color[3] = self.alpha
                        instances.append(_make_instance(pos, color, self.bs))
                    else:
                        color[3] = self.alpha * at
                        instances.append(_make_instance(pos, color, self.bs * (0.3 + 0.7 * at)))
            return np.array(instances, dtype=np.float32) if instances else np.zeros((0, 10), dtype=np.float32)

        # === Compute: element-by-element color transition (diagonal wave) ===
        if self.t > 0.0 or self.appear_t >= 1.0:
            ct = max(0.0, self.t)
            for r in range(rows):
                for c in range(cols):
                    elem_t = _wave_t(ct, r, c, rows, cols, 0.4)
                    pos = _box_pos(self.origin, r, c, self.sp)
                    color = self._colors_pre[r, c] * (1.0 - elem_t) + self._colors_post[r, c] * elem_t
                    color[3] = self.alpha
                    scale = self.bs
                    if 0.05 < elem_t < 0.95:
                        scale *= 1.0 + 0.2 * np.sin(elem_t * np.pi)
                    instances.append(_make_instance(pos, color, scale))

        # === Depart ===
        if self.depart_t > 0.0 and self.to_origin is not None:
            dt_val = ease_in_out_cubic(self.depart_t)
            for r in range(rows):
                for c in range(cols):
                    src = _box_pos(self.origin, r, c, self.sp)
                    dst = _box_pos(self.to_origin, r, c, self.sp)
                    pos = _lerp_pos(src, dst, dt_val)
                    color = self._colors_post[r, c].copy()
                    color[3] = self.alpha
                    instances.append(_make_instance(pos, color, self.bs))

        if not instances:
            return np.zeros((0, 10), dtype=np.float32)
        return np.array(instances, dtype=np.float32)


class StaticMatrixVisual:
    """Static matrix with flow support."""

    def __init__(self, matrix, origin, box_size=0.4, gap=0.1):
        self.matrix = matrix
        self.origin = np.array(origin, dtype=np.float32)
        self.bs = box_size
        self.sp = box_size + gap

        self.from_origin = None
        self.to_origin = None

        self.appear_t = 0.0
        self.t = 0.0
        self.depart_t = 0.0
        self.alpha = 1.0
        self.phase_group = 0
        self.is_stage_output = False
        self.output_alpha_mult = 1.0

        self._colors = None

    def _ensure_colors(self):
        if self._colors is None:
            self._colors = matrix_to_colors(self.matrix, 1.0)

    def get_instance_data(self):
        if self.alpha < 0.01 or self.output_alpha_mult < 0.01:
            return np.zeros((0, 10), dtype=np.float32)

        self._ensure_colors()
        rows, cols = self.matrix.shape
        instances = []

        if self.appear_t < 1.0:
            from_o = self.from_origin if self.from_origin is not None else self.origin
            seamless = self.from_origin is not None
            for r in range(rows):
                for c in range(cols):
                    at = _wave_t(self.appear_t, r, c, rows, cols)
                    src = _box_pos(from_o, r, c, self.sp)
                    dst = _box_pos(self.origin, r, c, self.sp)
                    pos = _lerp_pos(src, dst, at)
                    color = self._colors[r, c].copy()
                    if seamless:
                        color[3] = self.alpha
                        instances.append(_make_instance(pos, color, self.bs))
                    else:
                        color[3] = self.alpha * at
                        instances.append(_make_instance(pos, color, self.bs * (0.3 + 0.7 * at)))
        else:
            for r in range(rows):
                for c in range(cols):
                    pos = _box_pos(self.origin, r, c, self.sp)
                    color = self._colors[r, c].copy()
                    color[3] = self.alpha
                    instances.append(_make_instance(pos, color, self.bs))

        if self.depart_t > 0.0 and self.to_origin is not None:
            dt_val = ease_in_out_cubic(self.depart_t)
            for r in range(rows):
                for c in range(cols):
                    src = _box_pos(self.origin, r, c, self.sp)
                    dst = _box_pos(self.to_origin, r, c, self.sp)
                    pos = _lerp_pos(src, dst, dt_val)
                    color = self._colors[r, c].copy()
                    color[3] = self.alpha
                    instances.append(_make_instance(pos, color, self.bs))

        if not instances:
            return np.zeros((0, 10), dtype=np.float32)
        return np.array(instances, dtype=np.float32)
