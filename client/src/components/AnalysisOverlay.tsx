import React from "react";

type OverlayLayer = "Skeleton" | "Shuttle" | "Racket" | "Court map";

export type OverlayPoint = { x: number; y: number; confidence: number };

export type VerifiedOverlayFrame = {
  timeMs: number;
  errors?: Partial<Record<"Skeleton" | "Shuttle" | "Racket" | "Court map", string>>;
  skeleton?: OverlayPoint[];
  shuttle?: OverlayPoint;
  racket?: { grip: OverlayPoint; head: OverlayPoint };
  courtPosition?: { x: number; y: number; confidence: number };
};

type AnalysisOverlayProps = {
  frames: VerifiedOverlayFrame[];
  currentTimeMs: number;
  layers: Record<OverlayLayer, boolean>;
};

const skeletonEdges: Array<[number, number]> = [
  [5, 7], [7, 9], [6, 8], [8, 10], [5, 6], [5, 11], [6, 12], [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
];

export type OverlayLayerState = {
  key: OverlayLayer;
  label: string;
  status: "VERIFIED" | "UNCERTAIN" | "NO DATA" | "ERROR";
  error?: string;
};

export function getOverlayLayerStates(
  frame: VerifiedOverlayFrame,
  skeleton: OverlayPoint[],
  shuttle?: OverlayPoint,
  racket?: { grip: OverlayPoint; head: OverlayPoint },
  courtPosition?: OverlayPoint,
): OverlayLayerState[] {
  const layerStates: Array<{ key: OverlayLayer; label: string; ready: boolean; attempted: boolean; error?: string }> = [
    { key: "Skeleton", label: "POSE", ready: skeleton.some(isVisible), attempted: Boolean(frame.skeleton?.length), error: frame.errors?.Skeleton },
    { key: "Shuttle", label: "SHUTTLE", ready: isVisible(shuttle), attempted: Boolean(shuttle), error: frame.errors?.Shuttle },
    { key: "Racket", label: "RACKET", ready: Boolean(racket && isVisible(racket.grip) && isVisible(racket.head)), attempted: Boolean(racket), error: frame.errors?.Racket },
    { key: "Court map", label: "COURT", ready: isVisible(courtPosition), attempted: Boolean(courtPosition), error: frame.errors?.["Court map"] },
  ];
  return layerStates.map((layer) => ({
    key: layer.key,
    label: layer.label,
    error: layer.error,
    status: layer.error ? "ERROR" : layer.ready ? "VERIFIED" : layer.attempted ? "UNCERTAIN" : "NO DATA",
  }));
}

function nearestFrame(frames: VerifiedOverlayFrame[], targetTimeMs: number) {
  return frames.reduce<VerifiedOverlayFrame | undefined>((nearest, frame) => {
    if (!nearest || Math.abs(frame.timeMs - targetTimeMs) < Math.abs(nearest.timeMs - targetTimeMs)) return frame;
    return nearest;
  }, undefined);
}

function isVisible(point: OverlayPoint | undefined): point is OverlayPoint {
  return Boolean(point && point.confidence >= 0.45 && point.x >= 0 && point.x <= 100 && point.y >= 0 && point.y <= 100);
}

export function AnalysisOverlay({ frames, currentTimeMs, layers }: AnalysisOverlayProps) {
  const frame = nearestFrame(frames, currentTimeMs);
  if (!frame) return null;
  const skeleton = frame.skeleton ?? [];
  const racket = frame.racket;
  const shuttle = frame.shuttle;
  const courtPosition = frame.courtPosition;
  const layerStates = getOverlayLayerStates(frame, skeleton, shuttle, racket, courtPosition);

  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Verified frame analysis overlay">
      <defs>
        <filter id="verified-glow"><feGaussianBlur stdDeviation=".45" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter>
      </defs>
      {layers.Skeleton && <g filter="url(#verified-glow)" fill="none" stroke="#b7fa59" strokeLinecap="round" strokeWidth="0.42">
        {skeletonEdges.map(([first, second]) => {
          const start = skeleton[first];
          const end = skeleton[second];
          return isVisible(start) && isVisible(end) ? <line key={`${first}-${second}`} x1={start.x} y1={start.y} x2={end.x} y2={end.y} /> : null;
        })}
      </g>}
      {layers.Skeleton && skeleton.map((point, index) => isVisible(point) ? <circle key={index} cx={point.x} cy={point.y} r=".65" fill="#efffd4" stroke="#b7fa59" strokeWidth=".2" /> : null)}
      {layers.Racket && racket && isVisible(racket.grip) && isVisible(racket.head) && <g filter="url(#verified-glow)"><line x1={racket.grip.x} y1={racket.grip.y} x2={racket.head.x} y2={racket.head.y} stroke="#78e7f4" strokeLinecap="round" strokeWidth=".6" /><circle cx={racket.head.x} cy={racket.head.y} r=".7" fill="#78e7f4" /></g>}
      {layers.Shuttle && isVisible(shuttle) && <g filter="url(#verified-glow)"><circle cx={shuttle.x} cy={shuttle.y} r=".9" fill="#ffde73" /><circle cx={shuttle.x} cy={shuttle.y} r="1.8" fill="none" stroke="#ffde73" strokeOpacity=".55" strokeWidth=".18" /></g>}
      {layers["Court map"] && isVisible(courtPosition) && <g><circle cx={courtPosition.x} cy={courtPosition.y} r="1.2" fill="#b7fa59" fillOpacity=".18" stroke="#b7fa59" strokeWidth=".25" /><path d={`M ${courtPosition.x - 2.2} ${courtPosition.y} H ${courtPosition.x + 2.2} M ${courtPosition.x} ${courtPosition.y - 2.2} V ${courtPosition.y + 2.2}`} stroke="#b7fa59" strokeWidth=".25" /></g>}
      <g transform="translate(2 4)" fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace" fontSize="2.15" letterSpacing=".16">
        {layerStates.filter((layer) => layers[layer.key]).map((layer, index) => <g key={layer.key} transform={`translate(${index * 18} 0)`}><rect width="16" height="3.8" rx="1" fill="rgba(5,12,15,.82)" stroke={layer.status === "ERROR" ? "rgba(248,113,113,.85)" : layer.status === "VERIFIED" ? "rgba(183,250,89,.65)" : layer.status === "UNCERTAIN" ? "rgba(255,222,115,.75)" : "rgba(148,163,184,.4)"} strokeWidth=".18" /><circle cx="1.7" cy="1.9" r=".55" fill={layer.status === "ERROR" ? "#f87171" : layer.status === "VERIFIED" ? "#b7fa59" : layer.status === "UNCERTAIN" ? "#ffde73" : "#94a3b8"} /><text x="3" y="2.6" fill={layer.status === "ERROR" ? "#fecaca" : layer.status === "VERIFIED" ? "#d9ffad" : layer.status === "UNCERTAIN" ? "#ffefb4" : "#aab5bb"}>{`${layer.label} ${layer.status}`}</text></g>)}
      </g>
    </svg>
  );
}
