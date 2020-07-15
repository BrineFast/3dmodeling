import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import numpy as np
import glm

#Координаты вершин источника света
lightVertices = [
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,  0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,

        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,

        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,

         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,  1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,  1.0,  0.0,  0.0,
         0.5,  0.5,  0.5,  1.0,  0.0,  0.0,

        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,

        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5,  0.0,  1.0,  0.0
	]
lightCube = np.array(lightVertices, dtype= 'float32')

#Вершинный шейдер объекта
vertex_src = """
# version 330
layout(location = 0) in vec3 a_position;
layout(location = 2) in vec3 a_normal;
uniform mat4 model;
uniform mat4 projection;
uniform mat4 view;
uniform mat4 modelInv;
uniform mat4 modelView;

out vec3 Normal;
out vec3 fragPos;

void main()
{
    gl_Position = projection * view * model * vec4(a_position, 1.0);
    fragPos = vec3(modelView * vec4(a_position, 1.0f));
    Normal = mat3(modelInv) * a_normal;
}
"""

#Фрагментный шейдер объекта
fragment_src = """
#version 330 core

struct Light {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};

in vec3 fragPos;
in vec3 Normal;

out vec4 color;

uniform Material material;
uniform vec3 viewPos;
uniform Light light;

void main() {
 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - fragPos);
    float diff = max(dot(norm, lightDir), 0.0f);
    float spec;
    if (diff > 0.0){
        vec3 viewDir = normalize(viewPos - fragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        spec = pow(max(dot(viewDir, reflectDir), 0.0f), material.shininess);
    }
    else
        spec = 0.0;
    
    vec3 specular = light.specular * (spec * material.specular); 
    vec3 diffuse = light.diffuse * (diff * material.diffuse);
    vec3 ambient = light.ambient * material.ambient;
    
    vec3 result = ambient + diffuse + specular;
    
    color = vec4(result, 1.0f);
}
"""

#Вершинный шейдер источника света
vertex_light = """
# version 330
layout (location = 0) in vec3 a_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * vec4(a_position, 1.0);
}
"""

#Фрагментный шейдер источника света
fragment_light = """
#version 330 core
out vec4 color;
void main() {
	color = vec4(1.0f);
}
"""

#Класс для хранения значений параметров материала
class Material:
    def __init__(self):
        self.ambient = []
        self.diffuse = []
        self.specular = []
        self.shininess = []

#Класс объекта
class ObjLoader:
    def __init__(self):
        self.vert_coords = []
        self.text_coords = []
        self.norm_coords = []
        self.material = Material()

        self.model = []

    #Загрузка информации об объекте из файла
    def load_model(self):
        file_name = "car_triangles.txt"
        file = open(file_name, 'r')
        ambient = []
        diffuse = []
        text_coords = []
        specular = []
        figure = []
        shininess = 1.0
        flag = True
        while (flag):
            cmd = file.readline().split()
            if (cmd[0] == 'figure'):
                flag = False
            elif (cmd[0] == 'color'):
                ambient = [float(cmd[1]) / 255, float(cmd[2]) / 255, float(cmd[3]) / 255]
                diffuse = ambient
                specular = ambient
            elif (cmd[0] == 'ambient'):
                ambient = [float(cmd[1]), float(cmd[2]), float(cmd[3])]
            elif (cmd[0] == 'diffuse'):
                diffuse = [float(cmd[1]), float(cmd[2]), float(cmd[3])]
            elif (cmd[0] == 'specular'):
                specular = [float(cmd[1]), float(cmd[2]), float(cmd[3])]
            elif (cmd[0] == 'shininess'):
                shininess = float(cmd[1])
            elif (cmd[0] == 'mesh'):
                N = int(cmd[1])
                K = int(cmd[2])
                while (N > 0):
                    cmd1 = file.readline().split()
                    for i in range(6):
                        if i < 3:
                            self.vert_coords.append(float(cmd1[i]))
                        else:
                            self.norm_coords.append(float(cmd1[i]))
                    N -= 1
                while (K > 0):
                    cmd2 = file.readline().split()
                    for i in range(3):
                        text_coords.append(float(cmd2[i]))
                    K -= 1
                self.text_coords = np.array(text_coords, dtype = 'uint32')
                self.vert_coords = np.array(self.vert_coords, dtype = 'float32')
                self.norm_coords = np.array(self.norm_coords, dtype = 'float32')
                self.material.ambient = ambient
                self.material.diffuse = diffuse
                self.material.specular = specular
                self.material.shininess = shininess
                figure.extend(self.vert_coords)
                figure.extend(self.norm_coords)
                self.model = np.array(figure, dtype = 'float32')

#Функция изменения размеров окна
def window_resize(window, width, height):
    glViewport(0, 0, width, height)

#Проверка корректности инициализации библиотеки
if not glfw.init():
    raise Exception("glfw can not be initialized!")

#Создание окна
window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

#Проверка корректности создания окна
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

#Координаты камеры
global cameraXCoordinate
global cameraZCoordinate

cameraXCoordinate = 0
cameraZCoordinate = 25

#Управление камерой с помощью клавиатуры
def key_input(window, key, scancode, action, mode):
    global cameraXCoordinate, cameraZCoordinate
    if key == glfw.KEY_W and action == glfw.REPEAT:
        cameraZCoordinate -= 0.5
    if key == glfw.KEY_S and action == glfw.REPEAT:
        cameraZCoordinate += 0.5
    if key == glfw.KEY_A and action == glfw.REPEAT:
        cameraXCoordinate -= 0.5
    if key == glfw.KEY_D and action == glfw.REPEAT:
        cameraXCoordinate += 0.5

glfw.set_window_pos(window, 400, 200)
glfw.set_window_size_callback(window, window_resize)
glfw.set_key_callback(window, key_input)
glfw.make_context_current(window)
glEnable(GL_DEPTH_TEST)
glClearColor(0.3, 0.3, 0.3, 1.0)


#Компиляция шейдерных программ для объекта и источников света
shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))
lightShader = compileProgram(compileShader(vertex_light, GL_VERTEX_SHADER), compileShader(fragment_light, GL_FRAGMENT_SHADER))

#Загрузка объекта
obj = ObjLoader()
obj.load_model()

#Создание буферов и массива для объекта
objectVAO = glGenVertexArrays(1)
objectVBO = glGenBuffers(1)
objectEBO = glGenBuffers(1)

#Заполнение вершинного буфера
glBindBuffer(GL_ARRAY_BUFFER, objectVBO)
glBufferData(GL_ARRAY_BUFFER, obj.model.nbytes, obj.model, GL_STATIC_DRAW)

#Заполнение массива вершин
glBindVertexArray(objectVAO)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

#Заполнение массива индексов
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, obj.model.itemsize * 3, ctypes.c_void_p(20))
glEnableVertexAttribArray(1)

#Заполнение элементного буфера
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, objectEBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj.text_coords.nbytes, obj.text_coords, GL_STATIC_DRAW)
glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindVertexArray(0)

#Создание буфера и массива для источников света
lightCubeVAO = glGenVertexArrays(1)
lightCubeVBO = glGenBuffers(1)

#Заполнение вершинного буфера
glBindBuffer(GL_ARRAY_BUFFER, lightCubeVBO)
glBufferData(GL_ARRAY_BUFFER, lightCube.itemsize * 4, lightCube, GL_STATIC_DRAW)

#Заполнение массива вершин
glBindVertexArray(lightCubeVAO)
glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, lightCube.itemsize * 3, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

#Заполнение массива индексов
glVertexAttribPointer(1,3,GL_FLOAT,GL_FALSE, lightCube.itemsize * 3, ctypes.c_void_p(12))
glEnableVertexAttribArray(1)

#Создание массива для отображения блика света
lightVAO = glGenVertexArrays(1)

#Заполнение вершинного буфера
glBindBuffer(GL_ARRAY_BUFFER, lightCubeVBO)
glBufferData(GL_ARRAY_BUFFER, lightCube.nbytes, lightCube, GL_STATIC_DRAW)

#Заполнение массива вершин
glBindVertexArray(lightVAO)
glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, lightCube.itemsize * 6, ctypes.c_void_p(0))
glEnableVertexAttribArray(0)

projection = pyrr.matrix44.create_perspective_projection_matrix(90, 1280/720, 1, 2000) #Матрица проекции
translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0, 0, 0])) #Матрица преобразований
staticLightModel =  pyrr.matrix44.create_from_translation(pyrr.Vector3([7, 5, -4])) #Матрица преобразований статичного источника света

#Выбор используемого шейдера
glUseProgram(shader)

#Инициализация переменных для передачи данных в шейдерную программу
#***********************************************************************
lightPos = glGetUniformLocation(shader, "light.position")
lightAmbient = glGetUniformLocation(shader, "light.ambient")
lightDiffuse = glGetUniformLocation(shader, "light.diffuse")
lightSpecular = glGetUniformLocation(shader, "light.specular")
lightView = glGetUniformLocation(lightShader, "view")
lightProjection = glGetUniformLocation(lightShader, "projection")
lightModel = glGetUniformLocation(lightShader, "model")

materialAmbient = glGetUniformLocation(shader, "material.ambient")
materialDiffuse = glGetUniformLocation(shader, "material.diffuse")
materialSpecular = glGetUniformLocation(shader, "material.specular")
materialShinnes = glGetUniformLocation(shader, "material.shininess")

sceneModel = glGetUniformLocation(shader, "model")
sceneProjection = glGetUniformLocation(shader, "projection")
sceneView = glGetUniformLocation(shader, "view")
modelView = glGetUniformLocation(shader, "modelView")
modelInv = glGetUniformLocation(shader, "modelInv")
viewPos = glGetUniformLocation(shader, "viewPos")
#***********************************************************************

#Передача неизменяемых данных в шейдерную программу
#***********************************************************************
glUniformMatrix4fv(sceneProjection, 1, GL_FALSE, projection)
glUniform3fv(lightAmbient, 1, obj.material.ambient)
glUniform3fv(lightDiffuse, 1, obj.material.diffuse)
glUniform3fv(lightSpecular, 1, obj.material.specular)
glUniform3fv(materialAmbient, 1, obj.material.ambient)
glUniform3fv(materialDiffuse, 1, obj.material.diffuse)
glUniform3fv(materialSpecular, 1, obj.material.specular)
glUniform1f(materialShinnes, obj.material.shininess)
#***********************************************************************

#Цикл отрисовки изображения
while not glfw.window_should_close(window):
    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    #Матрица с положением камеры
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([cameraXCoordinate, 0, cameraZCoordinate]), pyrr.Vector3([cameraXCoordinate, 0, 0]), pyrr.Vector3([0, 1, 0]))  
    
    #Матрицы для движения источника света вокруг заданной точки
    move = pyrr.Matrix44.from_y_rotation(glfw.get_time() * 1)
    lightM = pyrr.matrix44.create_from_translation(pyrr.Vector3([15,0,5]))
    dynamicLightModel = move * lightM

    #Матрица отображения света
    inverseView = np.linalg.inv(view) * np.array(glm.vec4(0,0,0,1))

    #Матрица позиции света на объекте
    lightPosition = np.array(dynamicLightModel * staticLightModel * pyrr.Vector4([0, 0, 0, 1]))
    lightPosition = np.array([lightPosition[0],lightPosition[1],-lightPosition[2]])

    #Передача в шейдерную программу данных об положении камеры и света
    glUseProgram(shader)
    glUniform3fv(lightPos, 1, lightPosition)
    glUniform3fv(viewPos, 1, inverseView)
    glUniformMatrix4fv(sceneView, 1, GL_FALSE, view)

    #Матрица модели
    model = pyrr.matrix44.multiply(pyrr.Matrix44.from_y_rotation(3), translation)
    modeli = np.linalg.inv(model).transpose()

    #Передача в шейдерную программу данных об объекте
    glBindVertexArray(objectVAO)
    glUniformMatrix4fv(sceneModel, 1, GL_FALSE, model)
    glUniformMatrix4fv(modelView, 1, GL_FALSE, model)
    glUniformMatrix4fv(modelInv, 1, GL_FALSE, modeli)

    #Отрисовка объекта
    glDrawElements(GL_TRIANGLES, len(obj.text_coords), GL_UNSIGNED_INT, None)
    glBindVertexArray(0)

    #Отрисовка динамичного источника света
    glUseProgram(lightShader)
    glUniformMatrix4fv(lightView, 1, GL_FALSE, view)
    glUniformMatrix4fv(lightProjection, 1, GL_FALSE, projection)
    glUniformMatrix4fv(lightModel, 1, GL_FALSE, staticLightModel)
    glBindVertexArray(lightVAO)
    glDrawArrays(GL_TRIANGLES, 0, 36)
    glBindVertexArray(0)

    #Отрисовка динамичного источника света
    glUseProgram(lightShader)
    glUniformMatrix4fv(lightView, 1, GL_FALSE, view)
    glUniformMatrix4fv(lightProjection, 1, GL_FALSE, projection)
    glUniformMatrix4fv(lightModel, 1, GL_FALSE, dynamicLightModel)
    glBindVertexArray(lightVAO)
    glDrawArrays(GL_TRIANGLES, 0, 36)
    glBindVertexArray(0)

    glfw.swap_buffers(window)
glfw.terminate()