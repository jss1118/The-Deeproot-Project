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
    @Published var detections: [VNRecognizedObjectObservation] = []
    private var detectionRequest: VNCoreMLRequest?
    private let videoQueue = DispatchQueue(label: "videoQueue")
    
    // MARK: - Initialization
    override init() {
        super.init()
        setupSession()
        setupModel()
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
        session.addOutput(videoOutput)
    }
    
    /// Loads the CoreML object detection model and sets up the Vision request
    private func setupModel() {
        do {
            // Replace `MyObjectDetector` or `best()` with your actual CoreML model class or reference
            let mlModel = try VNCoreMLModel(for: best().model)
            detectionRequest = VNCoreMLRequest(model: mlModel, completionHandler: visionRequestDidComplete)
            detectionRequest?.imageCropAndScaleOption = .scaleFill
        } catch {
            print("Error setting up the CoreML model: \(error.localizedDescription)")
        }
    }
    
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
        guard let detectionRequest = detectionRequest,
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([detectionRequest])
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
        if let results = request.results as? [VNRecognizedObjectObservation] {
            print("Number of detections: \(results.count)")
            DispatchQueue.main.async {
                self.detections = results
            }
        }
    }
}

// MARK: - Camera Preview

/// A UIViewRepresentable that wraps an AVCaptureVideoPreviewLayer to display the live camera feed.
struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession

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
        return view
    }
    
    func updateUIView(_ uiView: VideoPreviewView, context: Context) {
        // No update needed
    }
}

// MARK: - Camera View

/// A SwiftUI view that displays the camera preview and overlays detection results.
struct CameraView: View {
    @StateObject private var cameraVM = CameraViewModel()
    
    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Display the live camera feed.
                CameraPreview(session: cameraVM.session)
                    .ignoresSafeArea()
                
                // Overlay bounding boxes for each detection
                ForEach(cameraVM.detections, id: \.uuid) { detection in
                    let bbox = detection.boundingBox
                    // Convert normalized bounding box [0..1] to the actual view coordinates
                    let boxWidth = geo.size.width * bbox.width
                    let boxHeight = geo.size.height * bbox.height
                    let xPos = geo.size.width * bbox.midX
                    // Flip y-axis because Vision’s origin is bottom-left,
                    // but SwiftUI’s origin is top-left
                    let yPos = geo.size.height * (1 - bbox.midY)
                    
                    Rectangle()
                        .stroke(Color.red, lineWidth: 2)
                        .frame(width: boxWidth, height: boxHeight)
                        .position(x: xPos, y: yPos)
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
                // Wrap MainView in a NavigationView so NavigationLink works properly.
                NavigationView {
                    MainView(loggedin: $loggedin, username: $username, password: $password)
                        .navigationBarHidden(true)
                }
            } else {
                ZStack {
                    // Background gradient
                    LinearGradient(gradient: Gradient(colors: [.blue.opacity(0.7), .green]),
                                   startPoint: .topLeading,
                                   endPoint: .bottomTrailing)
                        .ignoresSafeArea()

                    VStack(spacing: 20) {
                        Text("Deeproot AI")
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
                    // Reset username and password when the view disappears
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
            // Background gradient
            LinearGradient(gradient: Gradient(colors: [.blue.opacity(0.7), .green]),
                           startPoint: .topLeading,
                            endPoint: .bottomTrailing)
                .ignoresSafeArea()
            
            VStack {
                Spacer()
                
                // Center group: welcome text and diagnosis/crop buttons
                VStack(spacing: 200) {
                    // Centered welcome text
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
                    
                    // Diagnosis button and Crop type Menu
                    HStack(spacing: 40) {
                        Button(action: {
                            showCamera = true  // Set the state variable to true to show the camera sheet.
                        }) {
                            Label {
                                Text("Diagnosis")
                                    .font(.system(size: 20, weight: .bold, design: .default))
                                    .foregroundColor(.black)
                            } icon: {
                                Image(systemName: "camera")
                                    .font(.system(size: 40))
                            }
                            .padding()
                            .frame(width: 200, height: 200)
                            .background(LinearGradient(gradient: Gradient(colors: [.white, .blue]),
                                                       startPoint: .topLeading,
                                                       endPoint: .bottomTrailing))
                            .clipShape(Circle())
                        }
                        .sheet(isPresented: $showCamera) {
                            CameraView()
                        }

                        
                        Menu {
                            // Create menu items for each crop type.
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
                
                // Bottom buttons: Settings, Recents, Forums
                HStack(spacing: 16) {
                    NavigationLink(destination: SettingsView()) {
                        Label("Settings", systemImage: "gear")
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    
                    Button(action: {
                        print("Recents Button Pressed")
                    }) {
                        Label("Recents", systemImage: "camera.macro")
                            .foregroundColor(.white)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    
                    Button(action: {
                        print("Forums Button Pressed")
                    }) {
                        Label("Forums", systemImage: "bubble.right")
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
        // SettingsView is embedded in its own NavigationView so it has its own navigation bar.
        NavigationView {
            ZStack {
                Color(.systemGray6).ignoresSafeArea()
                
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

// MARK: - Preview
#Preview {
    LoginView(username: "", password: "")
}

