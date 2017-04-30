#include <cstdio>
#include "Headers.h"
#include <time.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>



void validate_errors(int line)
{
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
    {
        printf("OPENGL ERROR: FILE %s, LINE: %d", __FILE__, line);
        exit(-1);
    }

}

void validate_framebuffer_errors(int line)
{
    GLenum error = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER);
    if (error != GL_FRAMEBUFFER_COMPLETE)
    {
        printf("FRAMEBUFFER INCOMPLETE: FILE %s, LINE: %d", __FILE__, line);
        exit(-1);
    }

}

#define CHECK_ERROR validate_errors(__LINE__)
#define CHECK_FRAMEBUFFER validate_framebuffer_errors(__LINE__);

Shader PointShader; 
Shader SmoothShader;
Shader FinalShader;
Shader SKY_BOX_SHADER;
unsigned int mainScreenWidth, mainScreenHeight;
unsigned int textureWidth, textureHeight;
double reduction = 1.0;

Camera mainCamera(0, 0.9, 0.4, 0, 0, -0.1);
Camera texture_camera(0, 0, 0, 0, 0, -1);

void drawScene(void);
void initialize(void);

glm::vec3 quadCoord[] = {glm::vec3(-1, -1, -1), glm::vec3(1, -1, -1), glm::vec3(1, 1, -1), glm::vec3(-1, 1, -1)};
glm::vec2 quadTexCoord[] = {glm::vec2(0, 0), glm::vec2(1, 0), glm::vec2(1, 1), glm::vec2(0, 1)};

GLuint planeVertexBuffer = 0;
GLuint planeTexCoordBuffer = 0;

GLuint cubeMapHandler = 0;
byte* checkerTexture;

glm::vec3 deflector(0,0,0);
float deflectorRadius = 0.12;

int oldX = 0;
int oldY = 0;   

glm::vec3* skyBoxVertexArray;
unsigned int* skyBoxIndexArray;



bool rightClick = false;
bool renderSurface = false;


float size = 25;
particleInfo particleInformation;



SphContainer sphSolver(0, 0, 0, 0.7, 0.7, 0.7);

GLenum drbrf[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
glm::mat4 projection_matrix;
glm::mat4 model_matrix;
glm::mat4 prjMtr;
glm::mat4 modMtr;
glm::mat4 temp;
float matrForNormals[9];





void mouseClickHandler(int button, int state, int x, int y)
{
    oldX = x;
    oldY = y;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN)
        rightClick = true;
    if (button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
        rightClick = false;
};

void keyboardPressHandler(unsigned char key, int x, int y)
{
    if (key == 'r')
        renderSurface = !renderSurface;

	if (key == 's')
		sphSolver.createParticles(particleInformation);
}
	

struct ScreenSpaceRenderer
{
    unsigned int framebuffer;
    unsigned int depthBuffer;
    unsigned int depthTexture1;    
    unsigned int fluidDepthTexture1;
    unsigned int depthTexture2;    
    unsigned int fluidDepthTexture2;
    unsigned int normalTexture;

} RenderingSystem;


void setFramebufferOutputs();


void generateTextureBuffers()
{
    glEnable(GL_TEXTURE_2D);

    glGenTextures(1, &RenderingSystem.depthTexture1);
    glBindTexture(GL_TEXTURE_2D, RenderingSystem.depthTexture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glGenTextures(1, &RenderingSystem.depthTexture2);
    glBindTexture(GL_TEXTURE_2D, RenderingSystem.depthTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    glGenTextures(1, &RenderingSystem.fluidDepthTexture1);
    glBindTexture(GL_TEXTURE_2D, RenderingSystem.fluidDepthTexture1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


    glGenTextures(1, &RenderingSystem.fluidDepthTexture2);
    glBindTexture(GL_TEXTURE_2D, RenderingSystem.fluidDepthTexture2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, textureWidth, textureHeight, 0, GL_RED, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);

    CHECK_ERROR;

    glGenFramebuffersEXT(1, &RenderingSystem.framebuffer);

    glGenRenderbuffersEXT(1, &RenderingSystem.depthBuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER, RenderingSystem.depthBuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, textureWidth, textureHeight);
    glBindRenderbufferEXT(GL_RENDERBUFFER, 0);


    glBindFramebufferEXT(GL_FRAMEBUFFER, RenderingSystem.framebuffer);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderingSystem.depthBuffer);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, RenderingSystem.depthTexture2, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, RenderingSystem.fluidDepthTexture2, 0);
    CHECK_FRAMEBUFFER;
    glDrawBuffers(2, drbrf);
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
    CHECK_ERROR;
    glDisable(GL_TEXTURE_2D);

}

void setFramebufferOutputs()
{
    glBindFramebufferEXT(GL_FRAMEBUFFER, RenderingSystem.framebuffer);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, RenderingSystem.depthBuffer);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D, RenderingSystem.depthTexture2, 0);
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, RenderingSystem.fluidDepthTexture2, 0);

    CHECK_FRAMEBUFFER;

    glDrawBuffers(2, drbrf);
    glBindFramebufferEXT(GL_FRAMEBUFFER, 0);
}

void renderTextureOnScreen()
{

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glEnable(GL_TEXTURE_2D);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, RenderingSystem.depthTexture1);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, RenderingSystem.fluidDepthTexture1);

    glBindBuffer(GL_ARRAY_BUFFER, planeVertexBuffer);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, planeTexCoordBuffer);
    glTexCoordPointer(2, GL_FLOAT, 0, 0);
    glDrawArrays(GL_QUADS, 0, 4);     
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glActiveTexture(GL_TEXTURE0);
    glDisable(GL_TEXTURE_2D);
}

void deleteBuffers()
{
    glDeleteTextures(1, &RenderingSystem.fluidDepthTexture1);
    glDeleteTextures(1, &RenderingSystem.depthTexture1);
    glDeleteTextures(1, &RenderingSystem.fluidDepthTexture2);    
    glDeleteTextures(1, &RenderingSystem.depthTexture2);    
    glDeleteRenderbuffersEXT(1, &RenderingSystem.depthBuffer);
    glDeleteFramebuffersEXT(1, &RenderingSystem.framebuffer);
}

void changeSize(int w, int h) {

    mainScreenWidth = w;
    mainScreenHeight = h;    

    textureHeight = mainScreenHeight / reduction;
    textureWidth = mainScreenWidth / reduction;


    SmoothShader.sendFloat("texSizeX", 1.0f / textureWidth);
    SmoothShader.sendFloat("texSizeY", 1.0f / textureHeight);

    FinalShader.sendFloat("texSizeX", 1.0f / mainScreenWidth);
    FinalShader.sendFloat("texSizeY", 1.0f / mainScreenHeight);


    SmoothShader.sendInt("tSam", 0);
    SmoothShader.sendInt("deepSam", 1);

    FinalShader.sendInt("tSam", 0);
    FinalShader.sendInt("deepSam", 1);
    FinalShader.sendInt("envir", 2);

    glUseProgram(0);

    glViewport(0,0,mainScreenWidth,mainScreenHeight);    
    mainCamera.setAspectRatio((float)w / h);
    mainCamera.setRenderMatrix();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    deleteBuffers();
    generateTextureBuffers();
}



void mouseHandler(int x, int y)
{

    float dx = (oldX - x) / 60.0;
    float dy = (oldY - y) / 60.0;

    if (rightClick)
    {
        mainCamera.setRenderMatrix();
        glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(model_matrix));
        glGetFloatv(GL_PROJECTION_MATRIX, glm::value_ptr(projection_matrix));
        glm::vec4 defl = glm::vec4(deflector,1.0);
        defl.w = 1;
        glm::mat4 tmp = model_matrix * projection_matrix;
        glm::vec4 viewSpacePos = defl* tmp;
        viewSpacePos = viewSpacePos * (1.0f / viewSpacePos.w);
        viewSpacePos.x = (float)x / mainScreenWidth;
        viewSpacePos.x = viewSpacePos.x * 2 - 1;
        viewSpacePos.y = 1 - (float)y / mainScreenHeight;
        viewSpacePos.y = viewSpacePos.y * 2 - 1;
        viewSpacePos = viewSpacePos* glm::inverse(temp);
        viewSpacePos = viewSpacePos * (1.0f / viewSpacePos.w);
        glm::vec3 vel = glm::vec3(viewSpacePos) - deflector;
        deflector = glm::vec3(viewSpacePos);
        sphSolver.setPower(1, deflectorRadius, deflector, vel * 10.1f);
        oldX = x;
        oldY = y;
        return;
    }
    mainCamera.rotateAroundAim(dx, dy);
    oldX = x;
    oldY = y;   
    drawScene();
}

bool createWindow(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);

    mainScreenHeight = 1024;
    mainScreenWidth = 1024;
    glutInitWindowSize(mainScreenWidth,mainScreenHeight);
    textureHeight = mainScreenHeight / reduction;
    mainScreenWidth = mainScreenWidth / reduction;
    glutCreateWindow("Fluid simulation - SPH");
    glutDisplayFunc(drawScene);
    glutReshapeFunc(changeSize);
    glutMotionFunc(mouseHandler);
    glutMouseFunc(mouseClickHandler);
    glutKeyboardFunc(keyboardPressHandler);   
    glewInit();
    //generateCheckerTexture();
	
    texture_camera.setOrthoBase(1);
    glColor3f(1,1,1);
    glClearColor(0, 0, 0, 0);
    
    initialize();

	
    ///// CUDA accelerator initialization ////
    cudaDeviceProp properties;
    cudaSetDevice(0);
    cudaGLSetGLDevice(0);
    cudaGetDeviceProperties(&properties, 0);
   /* printf("CUDA Accelerator: %s \n", properties.name);
    printf("Maximum threads dim count: %d %d %d \n", properties.maxThreadsDim[0], properties.maxThreadsDim[1], properties.maxThreadsDim[2]);
    printf("Maximum grid size: %d %d %d \n", properties.maxGridSize[0], properties.maxGridSize[1], properties.maxGridSize[2]);
    printf("Maximum memory size: %d \n", properties.totalGlobalMem / 1024 / 1024);
    printf("Maximum threads per block: %d \n", properties.maxThreadsPerBlock);*/
    glutIdleFunc(drawScene);

    sphSolver.createParticles(particleInformation);
    glutMainLoop();

    return 0;
}

int main(int argc, char **argv)
{

    LOG_FILE_POINTER = stdout;
    texture_camera.setProjectionType(ORTHO_PROJECTION);
    particleInformation.particleCount = 70000;
    particleInformation.activeRadius = 0.024;
    particleInformation.fluidDensity = 1000.0f;
    particleInformation.fluidViscosity = 2.5f;
    particleInformation.stiffness = 2.5f;      

    createWindow(argc, argv);   
    return 0;
}   

void getMainMinor(glm::mat4 matrArr,float* arr)
{
	arr[0] = matrArr[0][0];
	arr[1] = matrArr[0][1];
	arr[2] = matrArr[0][2];

	arr[3] = matrArr[1][0];
	arr[4] = matrArr[1][1];
	arr[5] = matrArr[1][2];

	arr[6] = matrArr[2][0];
	arr[7] = matrArr[2][1];
	arr[8] = matrArr[2][2];

}
void initialize()
{

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_COLOR, GL_DST_COLOR);
    glBlendEquation(GL_FUNC_SUBTRACT);

    PointShader.createShader("shader/POINT_VX.vs", "shader/POINT_FS.fs", "");
    SmoothShader.createShader("shader/SMOOTH_VX.vs", "shader/SMOOTH_FS.fs", "");
    FinalShader.createShader("shader/FINAL_RENDER_SHADER_VS.vs", "shader/FINAL_RENDER_SHADER_FS.fs", "");
    SKY_BOX_SHADER.createShader("shader/sky_box_shader.vs", "shader/sky_box_shader.fs", "");

    glEnableVertexAttribArray(1);
    glGenBuffers(1, &planeVertexBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, planeVertexBuffer);
    glVertexAttribPointer(1, 3, GL_FLOAT, 0, 0, 0);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec3), quadCoord, GL_STATIC_DRAW);

    glEnableVertexAttribArray(2);
    glGenBuffers(1, &planeTexCoordBuffer);
    glBindBuffer(GL_ARRAY_BUFFER, planeTexCoordBuffer);
    glVertexAttribPointer(2, 2, GL_FLOAT, 0, 0, 0);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(glm::vec2), quadTexCoord, GL_STATIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, 0);  
    glPointSize(5);

    validate_errors;
}

void drawScene(void) {

    clock_t tm1 = clock();
    sphSolver.computeFluid(0.002);  




    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);   
    mainCamera.setRenderMatrix(); 
    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(model_matrix));
    glGetFloatv(GL_PROJECTION_MATRIX, glm::value_ptr(projection_matrix));   


    if (renderSurface)
    {
        glViewport(0, 0, textureWidth, textureHeight);
        PointShader.assignShader();    
        PointShader.sendViewMatrices(glm::value_ptr(projection_matrix), glm::value_ptr(model_matrix));
        std::swap(RenderingSystem.depthTexture1, RenderingSystem.depthTexture2);
        std::swap(RenderingSystem.fluidDepthTexture1, RenderingSystem.fluidDepthTexture2);
        setFramebufferOutputs();
        glBindFramebufferEXT(GL_FRAMEBUFFER, RenderingSystem.framebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
        PointShader.sendInt("drawingPass", 1);
    }

    sphSolver.drawParticles(); 

  

    if (renderSurface)
    {
        CHECK_ERROR;    
        SmoothShader.assignShader();
        std::swap(RenderingSystem.depthTexture1, RenderingSystem.depthTexture2);
        std::swap(RenderingSystem.fluidDepthTexture1, RenderingSystem.fluidDepthTexture2);
        setFramebufferOutputs();
        glBindFramebufferEXT(GL_FRAMEBUFFER, RenderingSystem.framebuffer);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        texture_camera.setRenderMatrix();
        glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modMtr));
        glGetFloatv(GL_PROJECTION_MATRIX, glm::value_ptr(prjMtr));
        SmoothShader.sendViewMatrices(glm::value_ptr(prjMtr), glm::value_ptr(modMtr));         

        renderTextureOnScreen();

        glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

        std::swap(RenderingSystem.depthTexture1, RenderingSystem.depthTexture2);
        std::swap(RenderingSystem.fluidDepthTexture1, RenderingSystem.fluidDepthTexture2);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);    
        glViewport(0, 0, mainScreenWidth, mainScreenHeight); 

        glEnable(GL_TEXTURE_CUBE_MAP);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapHandler);

        FinalShader.assignShader(); 
        FinalShader.sendViewMatrices(glm::value_ptr(prjMtr), glm::value_ptr(modMtr));
        FinalShader.sendMtr4x4Data("oldProjection_matrix", 1, glm::value_ptr(projection_matrix));
        FinalShader.sendMtr4x4Data("oldModelview_matrix", 1, glm::value_ptr(model_matrix));
		temp = glm::inverse(projection_matrix);
        FinalShader.sendMtr4x4Data("inverted_matrix", 1, glm::value_ptr(temp)); 

		temp = glm::inverse(model_matrix * projection_matrix);
        FinalShader.sendMtr4x4Data("toWorldMatrix", 1, glm::value_ptr(temp)); 

        temp = glm::inverse(model_matrix);
        getMainMinor(temp,matrForNormals);
		int loc = glGetUniformLocation(FinalShader.getProgram(), "normalToWorld");
		glUniformMatrix3fv(loc, 1, false, glm::value_ptr(glm::mat3(temp)));
		// FinalShader.sendMtr3x3Data("normalToWorld", 1, matrForNormals); 


        FinalShader.sendCameraPosition(&mainCamera);
        renderTextureOnScreen();
        glUseProgram(0);   
        glDisable(GL_TEXTURE_CUBE_MAP);
        glDisable(GL_TEXTURE_2D);
    }
    
    mainCamera.setRenderMatrix();
    glColor3f(1, 1, 1); 
    SKY_BOX_SHADER.assignShader();
    SKY_BOX_SHADER.sendViewMatrices(glm::value_ptr(projection_matrix), glm::value_ptr(model_matrix));

    glActiveTexture(GL_TEXTURE0);
    
    glUseProgram(0);

    sphSolver.drawContainer();


    temp = glm::translate(glm::mat4(1.0),deflector);

    glLoadIdentity();
    glTranslatef(deflector.x, deflector.y, deflector.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(temp));
    temp = temp * model_matrix;
    glLoadMatrixf(glm::value_ptr(temp));
    glColor3f(1, 0, 0);
    //glutWireSphere(deflectorRadius, 20, 20);
    glColor3f(1, 1, 1);

    glFlush();
    glutSwapBuffers();
    clock_t tm2 = clock();
    
}




