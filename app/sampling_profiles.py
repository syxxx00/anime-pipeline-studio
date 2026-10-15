"""Existing runtime defaults, not claims about each model's artistic role."""
PROFILES = {
    'newbie': dict(model_id='newbie', steps=28, cfg=4.5, sampler='res_multistep', scheduler='linear_quadratic', shift=6),
    'anima': dict(model_id='anima_turbo', steps=10, cfg=1, sampler='euler', scheduler='simple', shift=3),
    'illustrious': dict(model_id='illustrious', steps=28, cfg=5, sampler='dpmpp_2m_sde', scheduler='karras', shift=0),
    'z_image_turbo': dict(model_id='z_image_turbo', steps=9, cfg=1, sampler='euler', scheduler='simple', shift=3),
}
SOURCE_PROFILES = {'anima_newbie': 'anima', 'illustrious_newbie': 'illustrious', 'z_newbie': 'z_image_turbo'}
