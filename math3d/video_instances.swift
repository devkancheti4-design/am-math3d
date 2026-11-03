// VIDEO INSTANCES: Apple Vision lifts every foreground subject out of every frame of a video (foreground instance
// masks), on this Mac; nothing is sent anywhere.  One JSON line per frame:
// {"t": frame, "inst": [[x0, y0, x1, y1, area, cx, cy, footx, footy], ...]}  pixels of the video, y down;
// foot = the mean x of the instance's lowest 5% of rows, and its lowest row.
//     swiftc -O video_instances.swift -o video_instances && ./video_instances clip.mp4
import AVFoundation
import Foundation
import Vision

let path = CommandLine.arguments[1]
let asset = AVURLAsset(url: URL(fileURLWithPath: path))
let sem = DispatchSemaphore(value: 0)
var videoTrack: AVAssetTrack?
Task { videoTrack = try? await asset.loadTracks(withMediaType: .video).first; sem.signal() }
sem.wait()
guard let vt = videoTrack, let reader = try? AVAssetReader(asset: asset) else {
    FileHandle.standardError.write("cannot read \(path)\n".data(using: .utf8)!); exit(1)
}
let out = AVAssetReaderTrackOutput(track: vt, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
reader.add(out)
reader.startReading()
var t = 0
while let sample = out.copyNextSampleBuffer(), let pb = CMSampleBufferGetImageBuffer(sample) {
    let W = CVPixelBufferGetWidth(pb), H = CVPixelBufferGetHeight(pb)
    let handler = VNImageRequestHandler(cvPixelBuffer: pb, options: [:])
    let req = VNGenerateForegroundInstanceMaskRequest()
    var insts: [[Double]] = []
    if (try? handler.perform([req])) != nil, let obs = req.results?.first {
        for inst in obs.allInstances.sorted() {
            guard let mask = try? obs.generateScaledMaskForImage(forInstances: IndexSet(integer: inst), from: handler) else { continue }
            CVPixelBufferLockBaseAddress(mask, .readOnly)
            let mw = min(CVPixelBufferGetWidth(mask), W), mh = min(CVPixelBufferGetHeight(mask), H)
            let row = CVPixelBufferGetBytesPerRow(mask), fmt = CVPixelBufferGetPixelFormatType(mask)
            let base = CVPixelBufferGetBaseAddress(mask)!
            var rows = [Int](repeating: 0, count: mh), rowx = [Double](repeating: 0, count: mh)
            var x0 = W, y0 = H, x1 = -1, y1 = -1, area = 0, sx = 0.0, sy = 0.0
            for y in 0..<mh {
                for x in 0..<mw {
                    var on = false
                    if fmt == kCVPixelFormatType_OneComponent32Float {
                        on = base.advanced(by: y * row + x * 4).assumingMemoryBound(to: Float.self).pointee > 0.5
                    } else {
                        on = base.advanced(by: y * row + x).assumingMemoryBound(to: UInt8.self).pointee > 127
                    }
                    if on {
                        area += 1; sx += Double(x); sy += Double(y); rows[y] += 1; rowx[y] += Double(x)
                        x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x); y1 = max(y1, y)
                    }
                }
            }
            CVPixelBufferUnlockBaseAddress(mask, .readOnly)
            if area == 0 { continue }
            let cut = y1 - max(1, Int(Double(y1 - y0 + 1) * 0.05)) + 1
            var fx = 0.0, fn = 0
            for y in max(cut, 0)...y1 { fx += rowx[y]; fn += rows[y] }
            insts.append([Double(x0), Double(y0), Double(x1), Double(y1), Double(area), sx / Double(area), sy / Double(area),
                          fn > 0 ? fx / Double(fn) : sx / Double(area), Double(y1)])
        }
    }
    let line = try! JSONSerialization.data(withJSONObject: ["t": t, "inst": insts])
    print(String(data: line, encoding: .utf8)!)
    fflush(stdout)
    t += 1
}
