#include "Camera.h"
#include<glm/gtc/quaternion.hpp>

glm::vec3 xor (const glm::vec3 first, const glm::vec3 second)
{
	glm::vec3 returnVec;
	returnVec.x = first.y * second.z - first.z * second.y;
	returnVec.y = first.z * second.x - first.x * second.z;
	returnVec.z = first.x * second.y - first.y * second.x;
	return returnVec;
}

Camera::Camera(void)
{
    cameraWithAim = false;
    camera.x      = 0;
    camera.y      = 0;
    camera.z      = 0;
    aim.x         = 0;
    aim.y         = 0;
    aim.z         = 0;
    upVect.x      = 0;
    upVect.y      = 0;
    upVect.z      = 1;
    
    orthoBaseSize = 100;
    
    fov           = DEFAULT_FOV;
    aspectRatio   = DEFAULT_ASPECT_RATIO;
    zNear         = DEFAULTz_NEAR;
    zFar          = DEFAULTz_FAR;

    PROJECTION_TYPE = PERSPECTIVE_PROJECTION;
}

Camera::Camera(float x,float y,float z, float ax, float ay, float az)
{
    camera.x      = x;
    camera.y      = y;
    camera.z      = z;
    aim.x         = ax;
    aim.y         = ay;
    aim.z         = az;
    
    orthoBaseSize = 100;

    glm::vec3 dir =  aim - camera;
	glm::normalize(dir);

    glm::vec3 locUp = glm::vec3(0, 1, 0);
    if (abs(dir.z) < 0.999)
        locUp = glm::vec3(0, 0, 1);
    glm::vec3 locTang1 = xor(locUp , dir);
    upVect =  xor(dir , locTang1);

    fov           = DEFAULT_FOV;
    aspectRatio   = DEFAULT_ASPECT_RATIO;
    zNear         = DEFAULTz_NEAR;
    zFar          = DEFAULTz_FAR;
    PROJECTION_TYPE = PERSPECTIVE_PROJECTION;
}


void Camera::rotateAroundAim(float dx, float dy)
{
    glm::vec3 vertAxis(0,0,1);
    glm::vec3 vec = camera - aim;
    glm::quat cameraQuat(vec.x,vec.y,vec.z,0);
    glm::quat upVectQuat(upVect.x,upVect.y,upVect.z,0); 

	glm::rotate(cameraQuat, dx, vertAxis);
    //cameraQuat.rotateQuat(vertAxis, dx);    
	glm::rotate(upVectQuat, dx, vertAxis);
  //  upVectQuat.rotateQuat(vertAxis,dx);
	glm::vec3 const val1(cameraQuat.x, cameraQuat.y, cameraQuat.z);
	glm::vec3 const val2(upVectQuat.x, upVectQuat.y, upVect.z);
	glm::vec3 horizontalAxis = xor(val1 , val2);
   // cameraQuat.rotateQuat(horizontalAxis, dy);
	glm::rotate(cameraQuat, dy, horizontalAxis);
	glm::rotate(upVectQuat, dy, horizontalAxis);
    //upVectQuat.rotateQuat(horizontalAxis, dy);
    vec = glm::vec3(cameraQuat.x,cameraQuat.y,cameraQuat.z);
    upVect = xor(horizontalAxis,vec);
	glm::normalize(upVect);
    camera = vec  + aim; 
    return;
}

void Camera::rotateAroundCamera(const float dx,const float dy)
{
    glm::vec3 vertAxis(0,0,1);
    glm::vec3 aimVec = aim - camera;
    glm::quat aimQuat(aimVec.x,aimVec.y,aimVec.z, 0);
    glm::quat upVectQuat(vertAxis.x,vertAxis.y,vertAxis.z, 0);

	glm::rotate(aimQuat, dx, vertAxis);
	glm::rotate(upVectQuat, dx, vertAxis);
    //aimQuat.rotateQuat(vertAxis, dx);    
    //upVectQuat.rotateQuat(vertAxis,dx);
	const glm::vec3 val1(aimQuat.x, aimQuat.y, aimQuat.z);
	const glm::vec3 val2(upVectQuat.x, upVectQuat.y, upVectQuat.z);
	glm::vec3 horizontalAxis = xor (val1, val2);
  
	glm::rotate(aimQuat, dy, horizontalAxis);
	glm::rotate(upVectQuat, dy, horizontalAxis);
	// aimQuat.rotateQuat(horizontalAxis, dy);
   // upVectQuat.rotateQuat(horizontalAxis, dy);
    aimVec = glm::vec3(aimQuat.x,aimQuat.y,aimQuat.z);
    upVect = xor(horizontalAxis, aimVec);
    glm::normalize(upVect);   
    aim = aimVec + camera;
    return;
    
}

void Camera::dollyCam(float df)
{
    glm::vec3 vectorCam = camera - aim;
    vectorCam = vectorCam * (df);
    camera = aim + vectorCam;

}
void Camera::setRenderMatrix()
{
    glMatrixMode( GL_PROJECTION ); 
    glLoadIdentity(); 
    if (PROJECTION_TYPE == PERSPECTIVE_PROJECTION)
    {
        gluPerspective(fov,aspectRatio,zNear,zFar);   
    }
    else
    {
        glOrtho(-orthoBaseSize * aspectRatio, orthoBaseSize * aspectRatio, -orthoBaseSize, orthoBaseSize, zNear, zFar);
    }

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
    gluLookAt(camera.x,camera.y,camera.z,aim.x,aim.y,aim.z,upVect.x,upVect.y,upVect.z);     
    
    
    return;
}
void Camera::setAspectRatio(float newAscpect)
{
    aspectRatio = newAscpect;
}
void Camera::setFov(float newFOV)
{
    fov = newFOV;
}
Camera::~Camera(void)
{
}


glm::vec3 Camera::getAimPosition()
{
    return aim;
}
glm::vec3 Camera::getCameraPosition()
{
    return camera;
}

void Camera::setCameraPos(float x, float y, float z)
{
    camera = glm::vec3(x,y,z);
}

void Camera::setCameraPos(const glm::vec3& npos )
{
    camera = glm::vec3(npos.x,npos.y,npos.z);
}

void Camera::setAimPos(const glm::vec3& npos )
{
    aim = glm::vec3(npos.x,npos.y,npos.z);
}

void Camera::setAimPos(float x, float y, float z)
{   
    aim = glm::vec3(x,y,z);
}

void Camera::setAllPos(float x, float y, float z)
{   
    setCameraPos(x,y,z);
    setAimPos(x,y,z);
}

void Camera::setAllPos(const glm::vec3& npos)
{   
    setCameraPos(npos);
    setAimPos(npos);
}

void Camera::moveCamera(const glm::vec3& mvec)
{
    camera = camera + mvec;
}

void Camera::moveAim(const glm::vec3& mvec)
{
    aim = aim + mvec;
}

void Camera::setUpVec(float x, float y, float z)
{
    upVect.x = x;
    upVect.y = y;
    upVect.z = z;

}

void Camera::setProjectionType(int prjType)
{
    PROJECTION_TYPE = prjType;
}

void Camera::setOrthoBase(unsigned int base)
{
    orthoBaseSize = base;
}