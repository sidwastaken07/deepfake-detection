"""Regenerate docs/overlap_register.md from configs/roster.yaml."""

from indispoof.data.roster import DEFAULT_REGISTER, load_roster, render_register

if __name__ == "__main__":
    DEFAULT_REGISTER.write_text(render_register(load_roster()), encoding="utf-8")
    print(f"wrote {DEFAULT_REGISTER}")
