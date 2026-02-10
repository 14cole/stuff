import math
import unittest

import impedance_gui as gui
import impedance_sweep as sweep


class RegressionTests(unittest.TestCase):
    def _make_loaded_layer(self, eps: complex, mu: complex, thickness_in: float = 0.125) -> gui.LoadedLayer:
        table = gui.MaterialTable(
            freq_ghz=[1.0, 18.0],
            eps_r=[eps, eps],
            mu_r=[mu, mu],
        )
        return gui.LoadedLayer(
            thickness_m=thickness_in * gui.INCH_TO_M,
            anisotropic=False,
            polarization_deg=0.0,
            table_0deg=table,
            table_90deg=None,
        )

    def assertComplexAlmostEqual(self, a: complex, b: complex, tol: float = 1e-9) -> None:
        self.assertTrue(math.isclose(a.real, b.real, rel_tol=tol, abs_tol=tol), msg=f"real {a.real} != {b.real}")
        self.assertTrue(math.isclose(a.imag, b.imag, rel_tol=tol, abs_tol=tol), msg=f"imag {a.imag} != {b.imag}")

    def test_zero_thickness_matches_backing(self) -> None:
        z_air = sweep.compute_input_impedance(
            freq_hz=8.0e9,
            eps_r=3.2 + 0.5j,
            mu_r=1.3 + 0.2j,
            thickness_m=0.0,
            backing="air",
        )
        self.assertComplexAlmostEqual(z_air, sweep.ETA0 + 0.0j)

        z_pec = sweep.compute_input_impedance(
            freq_hz=8.0e9,
            eps_r=3.2 + 0.5j,
            mu_r=1.3 + 0.2j,
            thickness_m=0.0,
            backing="pec",
        )
        self.assertComplexAlmostEqual(z_pec, 0.0 + 0.0j)

    def test_air_matched_material_stays_matched(self) -> None:
        for freq_hz in (1.0e9, 6.5e9, 18.0e9):
            z = sweep.compute_input_impedance(
                freq_hz=freq_hz,
                eps_r=1.0 + 0.0j,
                mu_r=1.0 + 0.0j,
                thickness_m=0.014,
                backing="air",
            )
            self.assertComplexAlmostEqual(z, sweep.ETA0 + 0.0j)

    def test_gui_single_layer_matches_sweep_solver(self) -> None:
        layer = self._make_loaded_layer(eps=2.7 + 0.08j, mu=1.1 + 0.03j, thickness_in=0.145)
        f_ghz = 9.25

        z_gui = gui.compute_stack_impedance(f_ghz, [layer], "air")
        z_sweep = sweep.compute_input_impedance(
            freq_hz=f_ghz * sweep.GHZ_TO_HZ,
            eps_r=2.7 + 0.08j,
            mu_r=1.1 + 0.03j,
            thickness_m=layer.thickness_m,
            backing="air",
        )
        self.assertComplexAlmostEqual(z_gui, z_sweep)

    def test_vectorized_frequency_solver_matches_scalar(self) -> None:
        layer = self._make_loaded_layer(eps=4.1 + 0.12j, mu=1.4 + 0.06j, thickness_in=0.2)
        freqs = [1.0, 3.5, 7.0, 12.0, 18.0]

        z_many = gui.compute_stack_impedance_many(freqs, [layer], "pec")
        z_scalar = [gui.compute_stack_impedance(f, [layer], "pec") for f in freqs]

        for z0, z1 in zip(z_many, z_scalar):
            self.assertComplexAlmostEqual(z0, z1, tol=5e-9)

    def test_vectorized_angle_metrics_matches_scalar(self) -> None:
        layer = self._make_loaded_layer(eps=5.0 + 0.2j, mu=1.6 + 0.12j, thickness_in=0.1)
        freqs = [2.0, 6.0, 10.0]
        many = gui.compute_angle_metrics_many(freqs, 30.0, [layer], "te")

        for idx, f_ghz in enumerate(freqs):
            one = gui.compute_angle_metrics(f_ghz, 30.0, [layer], "te")
            for key in (
                "metal_loss_db",
                "metal_phase_deg",
                "air_loss_db",
                "air_phase_deg",
                "insertion_loss_db",
                "insertion_phase_deg",
            ):
                self.assertTrue(
                    math.isclose(many[key][idx], one[key], rel_tol=1e-7, abs_tol=1e-7),
                    msg=f"{key}: {many[key][idx]} != {one[key]}",
                )

    def test_uncertainty_scale_generation(self) -> None:
        disabled = gui.build_uncertainty_scales(
            gui.UncertaintyConfig(enabled=False, thickness_pct=5.0, eps_pct=5.0, mu_pct=5.0)
        )
        self.assertEqual(disabled, [(1.0, 1.0, 1.0)])

        enabled = gui.build_uncertainty_scales(
            gui.UncertaintyConfig(enabled=True, thickness_pct=5.0, eps_pct=5.0, mu_pct=0.0)
        )
        self.assertIn((1.0, 1.0, 1.0), enabled)
        self.assertGreaterEqual(len(enabled), 5)

    def test_nominal_uncertainty_scale_detection(self) -> None:
        self.assertTrue(gui.is_nominal_scale(1.0, 1.0, 1.0))
        self.assertTrue(gui.is_nominal_scale(1.0 + 1e-13, 1.0 - 1e-13, 1.0))
        self.assertFalse(gui.is_nominal_scale(1.01, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
