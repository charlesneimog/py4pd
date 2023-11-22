import pd
for library in ['torch', 'librosa']:
    try:
        exec('import ' + library)
    except:
        pd.error('Please install ' + library + ' library')
        import sys
        sys.exit(1)
        

def renderAudio_nn(audio, model):
    model = pd.get_patch_dir() + '/' + model
    audio = pd.get_patch_dir() + '/' +  audio
    torch.set_grad_enabled(False)
    model = torch.jit.load(model).eval()
    x = librosa.load(audio)[0]
    x_for = torch.from_numpy(x).reshape(1, 1, -1)
    z = model.encode(x_for)
    z[:, 0] += torch.linspace(-2, 2, z.shape[-1])
    y = model.decode(z).numpy().reshape(-1)
    pd.tabwrite('iaAudio', y.tolist(), resize=True)
    pd.print('Audio rendered')



def py4pdLoadObjects():
    pd.add_object(renderAudio_nn, "py4pd-nn")

