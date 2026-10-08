"""Mathematical and pipeline contracts; no real-data retraining."""
import os
from pathlib import Path
import tempfile
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
os.environ['KEOPS_CACHE_FOLDER'] = str(Path(tempfile.gettempdir()) / 'ustitch_test_keops')
os.environ['XDG_CACHE_HOME'] = str(Path(tempfile.gettempdir()) / 'ustitch_test_cache')
import numpy as np
import torch
from stitching_reconstruction.data import demo, prepare, load_visible, check_visible
from stitching_reconstruction.io import arrays, load_model
from stitching_reconstruction.train import initialize
from stitching_reconstruction.evaluate import trace
from stitching_reconstruction.physics.model import ScalarMLP, Stitching
from stitching_reconstruction.physics.shape_losses import ShapeStitching, configuration
torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        demo(self.root/'raw.npz')
        prepare(self.root/'raw.npz', self.root/'data', [.5])
        self.a, _ = load_visible(self.root/'data')

    def tearDown(self):
        self.tmp.cleanup()

    def model(self):
        init, _ = initialize(self.a, 8, 5, 17)
        torch.manual_seed(3)
        m = ShapeStitching(init['x'], init['lm'], init['grid'], init['bw'], init['obs'], init['targets'], configuration('baseline'), init['curve_kind'])
        return m, init

    def test_analytic_scalar_gradient(self):
        torch.manual_seed(3)
        net = ScalarMLP(3, (7, 7)).double()
        for layer in net.layers:
            layer.reset_parameters()
        x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        truth = torch.autograd.grad(net(x).sum(), x)[0]
        torch.testing.assert_close(net.gradient(x), truth, rtol=1e-11, atol=1e-12)

    def test_induced_reaction_continuity_identity(self):
        # d_t rho + div(rho*u) == rho*r, including heterogeneous masses.
        torch.manual_seed(4)
        m = Stitching(torch.randn(2, 4, 2), torch.zeros(2,4), [0.,1.], [.7,.9]).double()
        center = torch.randn(1,4,2,dtype=torch.float64)
        lm = torch.randn(1,4,dtype=torch.float64)
        xd = torch.randn_like(center); rate = torch.randn_like(lm)
        t = torch.tensor(.21,dtype=torch.float64,requires_grad=True)
        q = torch.randn(1,1,2,dtype=torch.float64,requires_grad=True)
        c, l = center+t*xd, lm+t*rate
        rho = torch.exp(torch.logsumexp(m.log_components(c,q)+l[:,None,:],-1))[0,0]
        u, r = m.induced_fields(c,l,xd,rate,q)
        dt = torch.autograd.grad(rho,t,create_graph=True)[0]
        div = sum(torch.autograd.grad((rho*u)[0,0,j],q,retain_graph=True)[0][0,0,j] for j in range(2))
        torch.testing.assert_close(dt+div,rho*r[0,0],rtol=1e-10,atol=1e-11)

    def test_train_archive_excludes_heldtime_and_truth(self):
        self.assertFalse(np.isin(.5,self.a['time']))
        self.assertNotIn('velocity_true_physical', self.a)
        (self.root/'data/test.npz').unlink()
        # Training data load works with no test file present.
        load_visible(self.root/'data')

    def test_split_overlap_rejected(self):
        broken = dict(self.a)
        broken['validation_rows'] = self.a['train_rows'][:3]
        with self.assertRaises(ValueError):
            check_visible(broken)

    def test_rollout_does_not_read_future_optimized_paths(self):
        m, _ = self.model()
        x, lm = m.trajectories[:1].detach().clone(), m.log_masses[:1].detach().clone()
        first = trace(m,x.clone(),lm.clone(),[0.,.3,.8],.05)
        with torch.no_grad():
            m.trajectories[1:].add_(100)
            m.raw_log_masses[1:].add_(10)
        second = trace(m,x.clone(),lm.clone(),[0.,.3,.8],.05)
        for a,b in zip(first,second):
            np.testing.assert_array_equal(a,b)

    def test_rollout_analytic_velocity_growth_and_knots(self):
        class Analytic:
            cfg = {'hard_mass':False}
            observed_times = torch.tensor([0.,.4,1.])
            def field(self,x,l,t,q):
                return torch.ones_like(q)*.7, torch.ones_like(l)*.2
        x,l = trace(Analytic(),torch.zeros(1,3,2),torch.zeros(1,3),[0.,.3,.9],.08)
        np.testing.assert_allclose(x[-1],.63,rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(l[-1],.18,rtol=1e-6,atol=1e-6)

    def test_checkpoint_roundtrip(self):
        m, init = self.model()
        path = self.root/'model.pt'
        torch.save(dict(state_dict=m.state_dict(),initial=init,config=m.cfg),path)
        loaded,_ = load_model(path,'cpu')
        x,l,t = m.trajectories[:1],m.log_masses[:1],m.grid[:1]
        for a,b in zip(m.field(x,l,t,x),loaded.field(x,l,t,x)):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_physical_time_mapping(self):
        a = arrays(self.root/'raw.npz')
        a['time'] = a['time']*24+12
        a['velocity_true'] = a['velocity_true']/24
        np.savez(self.root/'hours.npz',**a)
        prepare(self.root/'hours.npz',self.root/'hours',[24.])
        data,_ = load_visible(self.root/'hours')
        self.assertEqual(float(data['time_scale']),24.)
        self.assertEqual(float(data['time_offset']),12.)
        test = arrays(self.root/'hours/test.npz')
        np.testing.assert_allclose(test['velocity_true_physical'][0], np.array([.8,-.3])/24,rtol=1e-6)

    def test_two_visible_times(self):
        prepare(self.root/'raw.npz',self.root/'two',[.25,.5,.75])
        a,_ = load_visible(self.root/'two')
        init,curve = initialize(a,8,5,17)
        self.assertEqual(curve['selected'],'loglinear')
        self.assertTrue(np.isfinite(init['lm']).all())

    def test_no_zero_duration_after_float32_conversion(self):
        a = {k: v.copy() for k,v in self.a.items()}
        a['time'][a['time']==.25] += 1e-15
        a['observed_times'] = np.unique(a['time'])
        init,_ = initialize(a,8,5,17)
        self.assertTrue(np.all(np.diff(init['grid'])>0))


if __name__ == '__main__':
    unittest.main()
