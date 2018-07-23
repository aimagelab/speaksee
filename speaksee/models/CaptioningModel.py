import torch
from torch import nn

class CaptioningModel(nn.Module):
    def __init__(self):
        super(CaptioningModel, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def init_state(self, b_s, device):
        raise NotImplementedError

    def step(self, t, state, prev_output, images, seq=None, mode='teacher_forcing'):
        raise NotImplementedError

    def forward(self, images, seq):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, seq, mode='teacher_forcing')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return outputs

    def test(self, images, seq_len):
        device = images.device
        b_s = images.size(0)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, mode='test')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return torch.max(outputs, -1)[1]

    def beam_search(self, images, seq_len, eos_idx, beam_size, out_size=1):
        device = images.device
        b_s = images.size(0)
        state = self.init_state(b_s, device)
        outputs = []

        for i in range(b_s):
            state_i = tuple(s[i:i+1] for s in state)
            images_i = images[i:i+1]
            selected_words = None

            outputs_i = []
            logprobs_i = []
            tmp_outputs_i = [[] for _ in range(beam_size)]
            seq_logprob = .0
            for t in range(seq_len):
                word_logprob, state_i = self.step(t, state_i, selected_words, images_i, mode='beam_search')
                seq_logprob = seq_logprob + word_logprob
                selected_logprob, selected_idx = torch.sort(seq_logprob.view(-1), -1, descending=True)
                selected_logprob, selected_idx = selected_logprob[:beam_size], selected_idx[:beam_size]

                selected_beam = selected_idx / word_logprob.shape[1]
                selected_words = selected_idx - selected_beam*word_logprob.shape[1]

                # Update outputs with sequences that reached EOS
                outputs_i.extend([tmp_outputs_i[x.item()] for x in torch.masked_select(selected_beam, selected_words == eos_idx)])
                logprobs_i.extend([x.item() for x in torch.masked_select(selected_logprob, selected_words == eos_idx)])
                beam_size -= torch.sum(selected_words == eos_idx).item()

                # Remove sequence if it reaches EOS
                selected_beam = torch.masked_select(selected_beam, selected_words != eos_idx)
                selected_logprob = torch.masked_select(selected_logprob, selected_words != eos_idx)
                selected_words = torch.masked_select(selected_words, selected_words != eos_idx)

                tmp_outputs_i = [tmp_outputs_i[x.item()] for x in selected_beam]
                tmp_outputs_i = [o+[selected_words[x].item(),] for x, o in enumerate(tmp_outputs_i)]

                if selected_beam.shape[0] == 0:
                    break

                state_i = tuple(torch.index_select(s, 0, selected_beam) for s in state_i)
                seq_logprob = selected_logprob.view(-1, 1)

            # Update outputs with sequences that did not reach EOS
            outputs_i.extend(tmp_outputs_i)
            logprobs_i.extend([x.item() for x in selected_logprob])

            # Sort result
            outputs_i = [x for _,x in sorted(zip(logprobs_i,outputs_i), reverse=True)][:out_size]
            if len(outputs_i) == 1:
                outputs_i = outputs_i[0]
            outputs.append(outputs_i)

        return outputs