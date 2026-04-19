/**
 * Animated robot mascot — moods: idle, thinking, happy, sad.
 */

export type MascotMood = "idle" | "thinking" | "happy" | "sad";

type Props = {
  mood: MascotMood;
  /** Smaller variant for the typing row */
  size?: "header" | "inline";
};

export function PintuMascot({ mood, size = "header" }: Props) {
  const dim = size === "header" ? 52 : 36;
  const label =
    mood === "thinking"
      ? "Pintu is thinking"
      : mood === "happy"
        ? "Pintu is pleased"
        : mood === "sad"
          ? "Pintu could not answer"
          : "Pintu";

  return (
    <div
      className={`pintu-mascot pintu-mascot--${mood} pintu-mascot--${size}`}
      role="img"
      aria-label={label}
    >
      <svg
        width={dim}
        height={dim}
        viewBox="0 0 100 108"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        {/* Antenna */}
        <g className="pintu-mascot-antenna">
          <line x1="50" y1="8" x2="50" y2="22" stroke="url(#pg)" strokeWidth="4" strokeLinecap="round" />
          <circle cx="50" cy="6" r="5" fill="url(#pg)" className="pintu-mascot-antenna-ball" />
        </g>
        {/* Head */}
        <rect
          x="18"
          y="24"
          width="64"
          height="52"
          rx="18"
          fill="url(#pg)"
          stroke="rgba(255,255,255,0.25)"
          strokeWidth="2"
        />
        {/* Eyes */}
        <g className="pintu-mascot-eyes">
          <ellipse cx="38" cy="46" rx="9" ry="11" fill="#1e1b4b" className="pintu-mascot-eye pintu-mascot-eye-l" />
          <ellipse cx="62" cy="46" rx="9" ry="11" fill="#1e1b4b" className="pintu-mascot-eye pintu-mascot-eye-r" />
          <ellipse cx="40" cy="44" rx="3" ry="4" fill="white" opacity="0.85" />
          <ellipse cx="64" cy="44" rx="3" ry="4" fill="white" opacity="0.85" />
        </g>
        {/* Mouth — neutral / happy / sad via groups */}
        <g className="pintu-mascot-mouth-wrap">
          <path
            className="pintu-mascot-mouth pintu-mascot-mouth-idle"
            d="M 36 64 Q 50 70 64 64"
            stroke="#1e1b4b"
            strokeWidth="3"
            strokeLinecap="round"
            fill="none"
          />
          <path
            className="pintu-mascot-mouth pintu-mascot-mouth-happy"
            d="M 34 62 Q 50 78 66 62"
            stroke="#1e1b4b"
            strokeWidth="3"
            strokeLinecap="round"
            fill="none"
          />
          <path
            className="pintu-mascot-mouth pintu-mascot-mouth-sad"
            d="M 36 70 Q 50 58 64 70"
            stroke="#1e1b4b"
            strokeWidth="3"
            strokeLinecap="round"
            fill="none"
          />
        </g>
        {/* Body */}
        <rect
          x="26"
          y="78"
          width="48"
          height="28"
          rx="10"
          fill="url(#pg2)"
          stroke="rgba(255,255,255,0.2)"
          strokeWidth="2"
        />
        <circle cx="50" cy="92" r="4" fill="rgba(255,255,255,0.35)" className="pintu-mascot-core" />
        <defs>
          <linearGradient id="pg" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#c084fc" />
            <stop offset="100%" stopColor="#e879f9" />
          </linearGradient>
          <linearGradient id="pg2" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#a78bfa" />
            <stop offset="100%" stopColor="#7c3aed" />
          </linearGradient>
        </defs>
      </svg>
    </div>
  );
}
