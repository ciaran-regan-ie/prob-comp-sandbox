from hfppl  import Model, CachedCausalLM, Token, LMContext, smc_standard
from string import punctuation
import asyncio

class ConstraintModel(Model):
    def __init__(self, prompt, max_tokens, LLM):
        super().__init__()
        self.lm         = LMContext(LLM, prompt)
        self.q          = LMContext(LLM, prompt)
        self.prompt_len = len(str(self.lm.s))
        self.max_tokens = max_tokens

    async def step(self):
        # Which tokens are allowed?
        mask = self.active_constraint_mask()

        # Generate proposed token.
        token = await self.sample(self.lm.next_token(),
                                proposal = await self.proposal(mask))

        # Condition on constraint — a no-op since proposal already guarantees the constraint
        self.condition(token.token_id in mask)

        # Reduce number of max tokens remaining
        self.max_tokens -= 1

        print(str(self.lm.s)[self.prompt_len:])

        # Check if done
        if token == LLM.tokenizer.eos_token_id or self.max_tokens == 0:
            self.finish()

    def active_constraint_mask(self):
        string_so_far = str(self.lm.s)
        words = string_so_far.split()
        last_word = words[-1] if len(words) > 0 else ""
        return MASKS[min(5, len(last_word))]

    async def proposal(self, mask):
        string_so_far = str(self.lm.s)

        # Force the proposal StatefulLM to adhere to this mask
        await self.intervene(self.q.mask_dist(mask), True)

        # Return the proposal's modified next-token distribution
        return self.q.next_token()

if __name__ == '__main__':
    # Load the model
    print("Loading model...")
    LLM = CachedCausalLM.from_pretrained("lmsys/Vicuna-7b-v1.5", load_in_8bit = True)
    print("Model loaded.")

    # Define the masks
    MASKS = {i : set(j for (j,v) in enumerate(LLM.vocab)
                    if j != LLM.tokenizer.eos_token_id and '\n' not in v and
                    any(c.isalpha() or c in punctuation for c in v) and
                    len(v.strip()) <= 5 and (not v[0].isalpha() or i+len(v) <= 5))
                for i in range(6)}
    
    # From Politico.com
    prompt = """<|endoftext|>3 things to watch …

    1. The return of the House means new energy for the GOP’s Biden impeachment push, and Democrats are starting their pushback early. Rep. Jamie Raskin (D-Md.) is out this morning with a 14-page rebuttal memo that seeks to paint the GOP campaign as a “complete and total bust” and an attempt at distracting from the “overwhelming evidence of [Trump’s] criminal and corrupt conduct during his term of office.”

    2. The Senate is back this evening for a bed-check vote. With Minority Leader Mitch McConnell having successfully quieted (public) chatter about his health, expect senators to be quizzed anew about Sen. Tommy Tuberville’s (R-Ala.) Pentagon nominee blockade, especially with the Joint Chiefs chair, Gen. Mark Milley, just weeks away from retirement and the confirmation of his successor, Gen. C.Q. Brown, in limbo.

    3."""

    LLM.cache_kv(LLM.tokenizer.encode(prompt))

    async def run():
        constraint_model = ConstraintModel(prompt, 50, LLM)
        particles = await smc_standard(constraint_model, 20)
        for p in particles:
            print(str(p.lm.s)[p.prompt_len:])

    # Run the model
    asyncio.run(run())



