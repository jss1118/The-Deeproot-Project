import SwiftUI
import UIKit
import CoreML
import Vision
import AVFoundation

// MARK: - Camera ViewModel
class CameraViewModel: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    // MARK: - Properties
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    
    // Array of VNRequests, as recommended by Apple’s sample code
    private var requests = [VNRequest]()
    
    // Store the recognized observations so SwiftUI can display bounding boxes
    @Published var detections: [VNRecognizedObjectObservation] = []
    
    private let videoQueue = DispatchQueue(label: "videoQueue")
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupSession()
        setupVision()  // Apple’s recommended approach
    }
    
    /// Configures the capture session
    private func setupSession() {
        session.sessionPreset = .high
        
        // Configure input from the camera
        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device) else {
            print("Could not create video device input.")
            return
        }
        session.addInput(input)
        
        // Configure output to process video frames
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                // Rotation angles are specified in degrees:
                //  0   = portrait (no rotation)
                //  90  = landscapeLeft
                //  180 = portraitUpsideDown
                //  270 = landscapeRight
                connection.videoRotationAngle = 0  // Equivalent to .portrait
            } else {
                // Fallback for iOS < 17
                connection.videoOrientation = .portrait
            }
        }
    }
    
    /// Loads the CoreML model and creates a Vision request, storing it in the `requests` array
    private func setupVision() {
        do {
            // 1) Instantiate the model with thresholds:
            let config = MLModelConfiguration()
            let yoloModel = try yolocore(configuration: config)
           
            // 2) Create a Vision model from the instance’s .model:
            let visionModel = try VNCoreMLModel(for: yoloModel.model)
            
            // Create a Vision request with a completion handler.
            let objectRecognition = VNCoreMLRequest(model: visionModel) { [weak self] (request, error) in
                guard let self = self else { return }
                self.visionRequestDidComplete(request: request, error: error)
            }
            
            // 3) Use centerCrop option to match typical YOLO preprocessing
            objectRecognition.imageCropAndScaleOption = .centerCrop

            // Store the request
            requests = [objectRecognition]
            
        } catch {
            print("Error setting up the CoreML model: \(error.localizedDescription)")
        }
    }
    //h b
    /// Starts the capture session
    func startSession() {
        session.startRunning()
    }
    
    /// Stops the capture session
    func stopSession() {
        session.stopRunning()
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !requests.isEmpty,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        // Use .up orientation to match training, adjust if needed.
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer,
            orientation: .up,
            options: [:]
        )
        
        do {
            try handler.perform(requests)
        } catch {
            print("Failed to perform detection: \(error.localizedDescription)")
        }
    }
    
    // MARK: - Vision Completion Handler
    private func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let error = error {
            print("Vision error: \(error.localizedDescription)")
            return
        }
        
        guard let observations = request.results as? [VNRecognizedObjectObservation] else {
            print("No recognized objects returned.")
            return
        }
        
        // Filter out observations whose top label’s confidence is below 0.5:
        let confidenceThreshold: VNConfidence = 0.25
        let filtered = observations.filter { obs in
            guard let bestLabel = obs.labels.first else { return false }
            return bestLabel.confidence >= confidenceThreshold
        }
        
        // Log raw bounding box values for debugging.
        print("Total detections: \(observations.count). Kept after filtering: \(filtered.count)")
        for detection in filtered {
            if let topLabel = detection.labels.first {
                print("Detected: \(topLabel.identifier) [confidence: \(topLabel.confidence)] with bbox: \(detection.boundingBox)")
            }
        }
        
        DispatchQueue.main.async {
            self.detections = filtered
        }
    }
}

// MARK: - Camera Preview
/// A UIViewRepresentable that wraps an AVCaptureVideoPreviewLayer to display the live camera feed.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    var onPreviewLayerReady: ((AVCaptureVideoPreviewLayer) -> Void)? = nil

    class VideoPreviewView: UIView {
        override class var layerClass: AnyClass {
            AVCaptureVideoPreviewLayer.self
        }
        
        var videoPreviewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }
    }
    
    func makeUIView(context: Context) -> VideoPreviewView {
        let view = VideoPreviewView()
        view.videoPreviewLayer.session = session
        view.videoPreviewLayer.videoGravity = .resizeAspectFill
        
        // Pass the preview layer back so that bounding box conversion can be done
        DispatchQueue.main.async {
            self.onPreviewLayerReady?(view.videoPreviewLayer)
        }
        return view
    }
    
    func updateUIView(_ uiView: VideoPreviewView, context: Context) {
        // No update needed
    }
}

// MARK: - Camera View
/// A SwiftUI view that displays the camera preview and overlays detection results.
// MARK: - Camera View
/// A SwiftUI view that displays the camera preview and overlays detection results.
// MARK: - Camera View
/// A SwiftUI view that displays the camera preview and overlays detection results.
struct CameraView: View {
    @StateObject private var cameraVM = CameraViewModel()
    @State private var previewLayer: AVCaptureVideoPreviewLayer?
    
    var body: some View {
        GeometryReader { _ in
            ZStack {
                // Display the live camera feed
                CameraPreview(session: cameraVM.session, onPreviewLayerReady: { layer in
                    previewLayer = layer
                })
                .ignoresSafeArea()
                
                // Only draw bounding boxes if the preview layer is available.
                ForEach(cameraVM.detections, id: \.uuid) { detection in
                    if let layer = previewLayer {
                        // 1) Compute rect in a 'do' block (or local function)
                        let convertedRect: CGRect = {
                            var normRect = detection.boundingBox
                            normRect.origin.y = 1 - normRect.origin.y - normRect.height
                            return layer.layerRectConverted(fromMetadataOutputRect: normRect)
                        }()
                        
                        // 2) Return the actual view
                        Rectangle()
                            .stroke(Color.red, lineWidth: 2)
                            .frame(width: convertedRect.width, height: convertedRect.height)
                            .position(x: convertedRect.midX, y: convertedRect.midY)
                    }
                }

                
            }
            .onAppear {
                cameraVM.startSession()
            }
            .onDisappear {
                cameraVM.stopSession()
            }
        }
    }
}



// MARK: - Login & Main Views
var logran: Bool = false

@main
struct startup: App {
    var body: some Scene {
        WindowGroup {
            LoginView(username: "", password: "")
        }
    }
}

func getid() {
    if !logran {
        if let idForVendor = UIDevice.current.identifierForVendor?.uuidString {
            print("Vendor Identifier: \(idForVendor)")
            logran = true
        }
    }
}

struct LoginView: View {
    @State public var username: String
    @State public var password: String
    @State private var loggedin = false

    var body: some View {
        Group {
            if loggedin {
                NavigationView {
                    MainView(loggedin: $loggedin, username: $username, password: $password)
                        .navigationBarHidden(true)
                }
            } else {
                ZStack {
                    LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                                   startPoint: .topLeading,
                                   endPoint: .bottomTrailing)
                        .ignoresSafeArea()

                    VStack(spacing: 20) {
                        Text("Deeproot A.I")
                            .font(.largeTitle)
                            .fontWeight(.bold)
                            .foregroundColor(.white)

                        Text("Login")
                            .font(.title)
                            .fontWeight(.medium)
                            .foregroundColor(.white)

                        VStack(spacing: 15) {
                            TextField("Username", text: $username)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .padding()
                                .textInputAutocapitalization(.never)
                                .disableAutocorrection(true)

                            SecureField("Password", text: $password)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .padding()
                                .textInputAutocapitalization(.never)
                                .disableAutocorrection(true)
                        }
                        .background(RoundedRectangle(cornerRadius: 15)
                            .fill(Color.white.opacity(0.2)))
                        .padding()

                        Button(action: {
                            loggedin = true
                        }) {
                            Text("Login")
                                .font(.headline)
                                .foregroundColor(.white)
                                .padding()
                                .frame(maxWidth: .infinity)
                                .background(Color.blue)
                                .cornerRadius(10)
                                .padding(.horizontal)
                        }
                    }
                    .padding()
                }
                .onDisappear {
                    username = ""
                    password = ""
                }
            }
        }
    }
}

struct MainView: View {
    @Binding var loggedin: Bool
    @Binding var username: String
    @Binding var password: String

    // State variable to trigger the camera sheet.
    @State private var showCamera = false
    
    // Crop list declaration.
    let crop_type: [String] = [
        "Apple", "Casava", "Cherry", "Chili", "Citrus", "Coffee",
        "Corn", "Cucumber", "Grape", "Guava", "Jamun", "Lemon",
        "Mango", "Peach", "Pepper", "Pomegranate", "Potato", "Rice",
        "Soybean", "Strawberry", "Sugarcane", "Tea", "Tomato"
    ]
    @State private var model: String = ""
    @State private var selectedCrop: String = ""
    
    var body: some View {
        ZStack {
            LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                           startPoint: .topLeading,
                           endPoint: .bottomTrailing)
                .ignoresSafeArea()
            
            VStack {
                Spacer()
                
                VStack(spacing: 200) {
                    VStack {
                        Text("Welcome.")
                            .font(.system(size: 35, weight: .bold, design: .default))
                            .frame(width: 190, height: 40)
                            .padding()
                        
                        Text("Diagnose the root of disease with the click of a button")
                            .font(.system(size: 20, weight: .bold, design: .default))
                            .frame(width: 300, height: 80)
                            .multilineTextAlignment(.center)
                    }
                    
                    HStack(spacing: 40) {
                        Button(action: {
                            showCamera = true  // Show the camera sheet.
                        }) {
                            Label {
                                Text("Diagnosis")
                                    .font(.system(size: 20, weight: .bold, design: .default))
                                    .foregroundColor(.black)
                            } icon: {
                                Image(systemName: "camera")
                                    .font(.system(size: 40))
                            }
                            .frame(width: 200, height: 200)
                            .background(LinearGradient(gradient: Gradient(colors: [.white, .blue]),
                                                       startPoint: .topLeading,
                                                       endPoint: .bottomTrailing))
                            .clipShape(Circle())
                        }
                        .sheet(isPresented: $showCamera) {
                            CameraView()
                        }
                        .padding(.bottom, 100)
                        
                        Menu {
                            ForEach(crop_type, id: \.self) { type in
                                Button(action: {
                                    selectedCrop = type
                                    print(type)
                                    model = "model\(selectedCrop.lowercased())"
                                    print(model)
                                }) {
                                    Text(type)
                                }
                            }
                        } label: {
                            Text(selectedCrop.isEmpty ? "Crop Type" : selectedCrop)
                                .fontWeight(.bold)
                                .padding()
                                .background(Color.white)
                                .cornerRadius(5)
                        }
                        .padding(.bottom, 100)
                    }
                }
                
                Spacer()
                
                HStack(spacing: 50) {
                    NavigationLink(destination: SettingsView()) {
                        Label("Settings", systemImage: "gear")
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    
                    NavigationLink(destination: DeveloperView()) {
                        Label("Developer", systemImage: "hammer.circle")
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                }
                .padding(.bottom, 30)
            }
        }
        .onAppear {
            print("Logged in. Username: \(username), Password: \(password)")
        }
    }
}

struct SettingsView: View {
    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                               startPoint: .topLeading,
                               endPoint: .bottomTrailing).ignoresSafeArea()
                
                VStack(alignment: .leading, spacing: 20) {
                    Button(action: {
                        print("Model btn pressed")
                    }) {
                        Text("Models")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.gray)
                            .cornerRadius(10)
                            .padding(.horizontal, 20)
                    }
                    
                    Button(action: {
                        print("Theme btn pressed")
                    }) {
                        Text("Theme")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.gray)
                            .cornerRadius(10)
                            .padding(.horizontal, 20)
                    }
                    
                    Button(action: {
                        print("Beta btn pressed")
                    }) {
                        Text("Beta Development")
                            .font(.headline)
                            .foregroundColor(.white)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.gray)
                            .cornerRadius(10)
                            .padding(.horizontal, 20)
                    }
                    
                    Spacer()
                    
                    Image("logotransparent")
                        .resizable()
                        .frame(width: 400, height: 400)
                        .aspectRatio(contentMode: .fill)
                        .padding()
                    
                    Spacer()
                }
            }
            .navigationBarTitle("Settings", displayMode: .inline)
        }
    }
}

struct DeveloperView: View {
    @State private var devkey: String = ""
    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(gradient: Gradient(colors: [.gray.opacity(0.7), .black.opacity(0.65)]),
                               startPoint: .topLeading,
                               endPoint: .bottomTrailing)
                .ignoresSafeArea()
                
                VStack(spacing: 15) {
                    Text("Enter your developer key")
                        .font(.title)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                    Text("Get a developer key")
                        .font(.title3)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                        .frame(width: 400, height: 100, alignment: .center)
                        .padding(.bottom, 20)
                    
                    TextField("Developer key", text: $devkey)
                        .textFieldStyle(RoundedBorderTextFieldStyle())
                        .padding()
                        .textInputAutocapitalization(.never)
                        .disableAutocorrection(true)
                }
                .padding()
            }
        }
    }
}

// MARK: - Preview
#Preview {
    LoginView(username: "", password: "")
}

